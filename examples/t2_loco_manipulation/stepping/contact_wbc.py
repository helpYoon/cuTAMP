"""T2 contact-switching inverse-dynamics QP; world-frame sole wrenches.

Uses robot state only. Object contact forces are never read by this controller.
Torque is eliminated using the actuated rows of the dynamics equation.
"""
import time
import numpy as np
import mujoco
import proxsuite
from scipy.spatial.transform import Rotation


class ContactWholeBodyQP:
    def __init__(self, model, data, maps, trajectory):
        self.m, self.maps, self.ref = model, maps, trajectory
        if (model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
                or model.jnt_qposadr[0] != 0 or model.jnt_dofadr[0] != 0
                or model.body(int(model.jnt_bodyid[0])).name != 'trunk'):
            raise ValueError("T2 controller requires the robot free root as the first joint")
        if len(maps['active']) != 31 or len(maps.get('actuators', [])) != 31:
            raise ValueError("T2 controller requires 31 named body joints and actuators")
        self.failure_dump = None
        self.dofs = np.array(list(range(6))+[x[2] for x in maps['active']])
        self.qids = np.array([x[1] for x in maps['active']])
        self.feet = [model.body(n).id for n in ('left_ankle_roll_link','right_ankle_roll_link')]
        self.hands = [model.body(n).id for n in ('left_gripper/base','right_gripper/base')]
        self.sole = np.array([.042485,0.,-.032506])
        self.integral=np.zeros(31)
        self.foot_kp=180.
        self.foot_kd=28.
        self.posture_ki=30.
        self.integral_limit=.03
        self.friction_mu=1.
        self.friction_facets=4
        self.step_cop_inset=.015
        self.anchors = [data.xpos[i]+data.xmat[i].reshape(3,3)@self.sole for i in self.feet]
        self.rotations = [data.xmat[i].reshape(3,3).copy() for i in self.feet]
        self.mass = np.zeros((model.nv,model.nv))
        self.com_anchor=data.subtree_com[model.jnt_bodyid[0]].copy()
        self.scratch=mujoco.MjData(model)
        self.reference_data=mujoco.MjData(model)
        self.solver = None
        self.timings = []
        self.cpu_timings = []
        self.previous_j = None
        self.last_tau = np.zeros(31)

    def jac(self, data, body, point):
        p,r = np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv))
        mujoco.mj_jac(self.m,data,p,r,point,body)
        return np.vstack((p[:,self.dofs],r[:,self.dofs]))

    def step(self, data, t, *, state_is_current=False):
        start=time.perf_counter()
        cpu_start=time.thread_time()
        if not state_is_current:mujoco.mj_forward(self.m,data)
        packet=self.ref.packet(t) if hasattr(self.ref,'packet') else None
        x=packet[0] if packet is not None else self.ref.sample(np.array([t]))[0]
        mujoco.mj_fullM(self.m,self.mass,data.qM)
        M=self.mass[np.ix_(self.dofs,self.dofs)]
        h=(data.qfrc_bias-data.qfrc_passive)[self.dofs].copy()
        v=data.qvel[self.dofs]
        self.integral=np.clip(self.integral+.002*(x[6:37]-data.qpos[self.qids]),-self.integral_limit,self.integral_limit)
        h[6:]+=self.m.dof_frictionloss[self.dofs[6:]]*np.tanh(v[6:]/.01)
        feet_packet=self.ref.feet_packet(t) if hasattr(self.ref,"feet_packet") else None
        contacts=np.ones(2,dtype=bool) if feet_packet is None else feet_packet[3]
        js=[]; errors=[]
        for i,b in enumerate(self.feet):
            R=data.xmat[b].reshape(3,3); p=data.xpos[b]+R@self.sole
            js.append(self.jac(data,b,p))
            target_p=self.anchors[i] if feet_packet is None else feet_packet[0][i,:3,3]
            target_R=self.rotations[i] if feet_packet is None else feet_packet[0][i,:3,:3]
            errors.extend(np.r_[target_p-p,Rotation.from_matrix(target_R@R.T).as_rotvec()])
        J=np.vstack(js)
        # Jdot*v by a forward kinematic directional derivative, not time history.
        scratch=self.scratch;scratch.qpos[:]=data.qpos
        mujoco.mj_integratePos(self.m,scratch.qpos,data.qvel,1e-6)
        mujoco.mj_kinematics(self.m,scratch);mujoco.mj_comPos(self.m,scratch)
        Jnext=np.vstack([self.jac(scratch,b,scratch.xpos[b]+scratch.xmat[b].reshape(3,3)@self.sole) for b in self.feet])
        jd=(Jnext-J)@v/1e-6
        if hasattr(self,'payload_rows'):
            for i,b in enumerate(self.hands):
                load=self.payload_rows[i];Rhand=data.xmat[b].reshape(3,3)
                f=np.array([0.,0.,-9.81*load[0]])
                moment=np.cross(Rhand@load[1:],np.array([0.,0.,-9.81]))
                h-=self.jac(data,b,data.xpos[b]).T@np.r_[f,moment]
        D=np.column_stack((M,-J.T))
        # qdd(37), left/right sole wrench xyz force / xyz moment (12).
        H=np.eye(49)*1e-4;g=np.zeros(49)
        def task(A,b,w):
            nonlocal H,g
            H+=w*(A.T@A);g-=w*(A.T@b)
        qerr=x[6:37]-data.qpos[self.qids]
        if packet is None:
            xp=self.ref.sample(np.array([t+.001]))[0];xm=self.ref.sample(np.array([t-.001]))[0]
        desired_acc=packet[1] if packet is not None else (xp[43:74]-xm[43:74])/.002
        # Integrate in acceleration units; inverse dynamics supplies inertia scaling.
        # A fixed torque integral destabilizes T2 wrist/head joints with small inertia.
        target=desired_acc+80*qerr+18*(x[43:74]-v[6:])+self.posture_ki*self.integral
        A=np.zeros((31,49));A[:,6:37]=np.eye(31);task(A,target,1.)
        # Base translation and orientation stabilize the floating trunk.
        R=data.xmat[self.m.jnt_bodyid[0]].reshape(3,3)
        Rref=packet[5] if packet is not None else Rotation.from_euler('ZYX',x[3:6]).as_matrix()
        if packet is not None and len(packet)>7:
            omega_ref=R.T@packet[7]
            alpha_ref=R.T@packet[8]-np.cross(v[3:6],omega_ref)
            base=np.r_[packet[9]+60*(x[:3]-data.qpos[:3])+16*(x[37:40]-v[:3]),alpha_ref+60*(R.T@Rotation.from_matrix(Rref@R.T).as_rotvec())+16*(omega_ref-v[3:6])]
        else:
            base=np.r_[60*(x[:3]-data.qpos[:3])-16*v[:3],60*(R.T@Rotation.from_matrix(Rref@R.T).as_rotvec())-16*v[3:6]]
        A=np.zeros((6,49));A[:,:6]=np.eye(6);task(A,base,1.)
        jc=np.zeros((3,self.m.nv));mujoco.mj_jacSubtreeCom(self.m,data,jc,int(self.m.jnt_bodyid[0]))
        jc=jc[:,self.dofs];A=np.zeros((3,49));A[:,:37]=jc
        rd=self.reference_data;rd.qpos[:]=data.qpos;rd.qpos[:3]=x[:3]
        if packet is not None:rd.qpos[3:7]=packet[6]
        else:
            quat=Rotation.from_euler('ZYX',x[3:6]).as_quat();rd.qpos[3:7]=quat[[3,0,1,2]]
        rd.qpos[self.qids]=x[6:37];mujoco.mj_kinematics(self.m,rd);mujoco.mj_comPos(self.m,rd)
        task(A,100*(rd.subtree_com[self.m.jnt_bodyid[0]]-data.subtree_com[self.m.jnt_bodyid[0]])-25*(jc@v),300.)
        hands=packet[2] if packet is not None else self.ref.sample_hands(np.array([t]))[0]
        if packet is None:hm=self.ref.sample_hands(np.array([t-.001]))[0];hp=self.ref.sample_hands(np.array([t+.001]))[0]
        for i,b in enumerate(self.hands):
            j=self.jac(data,b,data.xpos[b]);rr=data.xmat[b].reshape(3,3)
            err=np.r_[hands[i,:3]-data.xpos[b],Rotation.from_matrix(hands[i,3:12].reshape(3,3)@rr.T).as_rotvec()]
            A=np.zeros((6,49));A[:,:37]=j
            if packet is not None:vel=packet[3][i];acc=packet[4][i]
            else:
                vel=np.r_[(hp[i,:3]-hm[i,:3])/.002,Rotation.from_matrix(hp[i,3:12].reshape(3,3)@hm[i,3:12].reshape(3,3).T).as_rotvec()/.002]
                acc=np.r_[(hp[i,:3]-2*hands[i,:3]+hm[i,:3])/1e-6,np.zeros(3)]
            jnext=self.jac(scratch,b,scratch.xpos[b]);jdotv=(jnext-j)@v/1e-6
            task(A,acc+400*err+40*(vel-j@v)-jdotv,100.)
        footA=np.column_stack((J,np.zeros((12,12))))
        fv=np.zeros(12) if feet_packet is None else feet_packet[1].reshape(-1)
        fa=np.zeros(12) if feet_packet is None else feet_packet[2].reshape(-1)
        footb=fa+self.foot_kp*np.array(errors)+self.foot_kd*(fv-J@v)-jd
        task(footA,footb,1000.)
        contactA=footA.copy();contactb=footb.copy()
        for i,active_contact in enumerate(contacts):
            if not active_contact:
                contactA[6*i:6*i+6]=0.;contactA[6*i:6*i+6,37+6*i:43+6*i]=np.eye(6);contactb[6*i:6*i+6]=0.
        rows=[D[:6],contactA,D[6:]]
        low=[-h[:6],contactb]
        high=[low[0].copy(),contactb.copy()]
        acts=np.array(self.maps['actuators']); limits=self.m.actuator_ctrlrange[acts]
        low.append(limits[:,0]-h[6:]);high.append(limits[:,1]-h[6:])
        # Wrench cone, conservative rectangular CoP and torsional friction.
        step_inset=self.step_cop_inset if feet_packet is not None and (not bool(np.all(contacts)) or float(np.min(feet_packet[4]))<999.) else 0.
        for foot in range(2):
            if self.friction_facets>4:
                for angle in np.arange(self.friction_facets)*2*np.pi/self.friction_facets:
                    a=np.zeros((1,49));a[0,37+foot*6]=np.cos(angle);a[0,38+foot*6]=np.sin(angle);a[0,39+foot*6]=-self.friction_mu*np.cos(np.pi/self.friction_facets)
                    rows.append(a);low.append(np.array([-np.inf]));high.append(np.array([0.]))
                components=[(3,.040886-step_inset),(4,.10791-step_inset),(5,.005)]
            else:components=[(0,self.friction_mu),(1,self.friction_mu),(3,.040886-step_inset),(4,.10791-step_inset),(5,.005)]
            for axis,scale in components:
                for sign in [-1,1]:
                    a=np.zeros((1,49));a[0,37+foot*6+axis]=sign;a[0,39+foot*6]=-scale
                    rows.append(a);low.append(np.array([-np.inf]));high.append(np.array([0.]))
            a=np.zeros((1,49));a[0,39+foot*6]=1
            rows.append(a);low.append(np.array([0.]));high.append(np.array([1000. if feet_packet is None else max(.001,feet_packet[4][foot])]))
        # Joint limit recovery and one-step velocity viability bounds.
        a=np.zeros((31,49));a[:,6:37]=np.eye(31)
        ranges=self.m.jnt_range[[i[0] for i in self.maps['active']]]
        lo=np.maximum((-12-v[6:])/.002,2*(ranges[:,0]-data.qpos[self.qids]-.02*v[6:])/.02**2)
        hi=np.minimum((12-v[6:])/.002,2*(ranges[:,1]-data.qpos[self.qids]-.02*v[6:])/.02**2)
        rows.append(a);low.append(lo);high.append(hi)
        dense=np.vstack(rows);l=np.concatenate(low);u=np.concatenate(high)
        scale=max(1.,float(np.max(np.abs(H))))
        H=H/scale;g=g/scale
        eq=l==u
        if self.solver is None:
            self.solver=proxsuite.proxqp.dense.QP(49,int(eq.sum()),int((~eq).sum()))
            self.solver.settings.eps_abs=1e-6;self.solver.settings.eps_rel=1e-6
            self.solver.settings.max_iter=1000
            self.solver.settings.eps_primal_inf=1e-12;self.solver.settings.eps_dual_inf=1e-12
            self.solver.init(H,g,dense[eq],l[eq],dense[~eq],l[~eq],u[~eq])
        else:self.solver.update(H=H,g=g,A=dense[eq],b=l[eq],C=dense[~eq],l=l[~eq],u=u[~eq])
        self.solver.solve();result=self.solver.results
        z=result.x
        violation=max(float(np.max(l-dense@z)),float(np.max(dense@z-u)))
        if violation>1e-3 or not np.isfinite(z).all():
            if self.failure_dump is not None:
                np.savez(self.failure_dump,H=H,g=g,A=dense,l=l,u=u,qpos=data.qpos,qvel=data.qvel,solution=z,time=t)
            raise RuntimeError('QP constraint residual '+str(violation))
        z=result.x; tau=D[6:]@z+h[6:]
        residual=float(np.max(np.abs(D[:6]@z+h[:6])))
        if not np.isfinite(tau).all() or residual>.02:raise RuntimeError('invalid dynamics solution')
        self.cpu_timings.append((time.thread_time()-cpu_start)*1000)
        self.timings.append((time.perf_counter()-start)*1000)
        self.last_wrenches=z[37:].reshape(2,6).copy()
        self.last_tau=tau
        self.last_contacts=contacts.copy()
        return tau
