            subroutine tcteos(temp,den,ye, 
           1                     ptot,etot,stot,dpdd,dpdt,dedd,dedt,dsdd,dsdt) 
            implicit none
            save  
c.. 
c..this routine performs a thermodynamically consistent interpolation 
c..in a tabular electron-positron equation of state using biquintic 
c..hermite basis functions. 
c.. 
c..input : 
c..temp = temperature(in K) 
c..den = density (in g cm~3) 
c..ye = electrons per baryon = zbar/abar 
c.. 
c..also input through a common block is the table of the helmholtz 
c..free energy and eight of its partial derivatives : 
c..f, df–d, df–t, df–dd, df–tt, df–dt, df–ddt, df–dtt and df–ddtt  
c.. 
c..output : 
c..ptot = pressure (ergs cm~3) 
c..etot = specific internal energy (ergs g~1) 
c..stot = specific entropy (ergs g~1 K~1) 
c..dpdd = partial derivative of pressure with density (ergs g~1) 
c..dpdt = partial derivative of pressure with temperature (ergs cm~3 K~1) 
c..dedd = partial derivative of energy with density (ergs cm~3 g~2) 
c..dedt = partial derivative of energy with temperature (ergs g~1 K~1) 
c..dsdd = partial derivative of entropy with density (ergs cm~3 g~2 K~1)  
c..dsdt = partial derivative of entropy with temperature (ergs g~1 K~2) 
c.. 
c..declare the pass  
        double precision temp,den,ye,ptot,etot,stot,dpdd,dpdt,dedd,dedt,
       1 dsdd,dsdt 
c..declare the internal variables  
        integer i,j,iat,jat 
        double precision tlo,thi,tstp,tstpi,dlo,dhi,dstp,dstpi, 
       1 tsav,dsav,free,df–d,df–t,df–dd,df–tt,df–dt,dt,dt2,dti,dt2i,dd,  
       2 dd2,ddi,dd2i,xt,xd,mxt,mxd,si0t,si1t,si2t,si0mt,si1mt,si2mt,si0d, 
       3 si1d,si2d,si0md,si1md,si2md,dsi0t,dsi1t,dsi2t,dsi0mt,dsi1mt, 
       4 dsi2mt,dsi0d,dsi1d,dsi2d,dsi0md,dsi1md,dsi2md,ddsi0t,ddsi1t, 
       5 ddsi2t,ddsi0mt,ddsi1mt,ddsi2mt,ddsi0d,ddsi1d,ddsi2d,ddsi0md, 
       6 ddsi1md,ddsi2md,z,psi0,dpsi0,ddpsi0,psi1,dpsi1,ddpsi1,psi2,dpsi2, 
       7 ddpsi2,w0t,w1t,w2t,w0mt,w1mt,w2mt,w0d,w1d,w2d,w0md,w1md,w2md, 
       8 din,herm5