#include "../src/kernel.cuh"
#include "../src/rbcStatistics.cuh"
#include <cassert>
#include <iostream>
#include <random>

static void close(double a,double b,double tol=5.e-7)
{
    if(std::abs(a-b)>tol) {std::cerr<<"mismatch "<<a<<" vs "<<b<<'\n';std::exit(1);}
}
static int pairIndex(int i,int j) {return i==j ? i : i+j==1 ? 3 : i+j==2 ? 4 : 5;}
static void quadrature()
{
    std::mt19937 gen(123);std::uniform_real_distribution<float> uni(-0.03,0.03);
    for(int trial=0;trial<100;++trial)
    {
        HydroState s{};s.rho=0.9+uni(gen);
        real_t a[3];for(int i=0;i<3;++i) {s.v[i]=uni(gen);a[i]=uni(gen)*0.4;}
        for(int i=0;i<6;++i) s.p[i]=uni(gen)*0.01;
        double rho=0,j[3]={},p[6]={},a3[7]={},rho2=0,j2[3]={},p2[6]={};
        constexpr_for<0,27>([&](auto q) {
            constexpr int c[3]={D3Q27::cx<q>(),D3Q27::cy<q>(),D3Q27::cz<q>()};
            const double f=hydroPopulation<q,3>(s,a),f2=hydroPopulation<q,2>(s,a);
            rho+=f;rho2+=f2;
            for(int i=0;i<3;++i) {j[i]+=c[i]*f;j2[i]+=c[i]*f2;}
            for(int i=0;i<3;++i) for(int k=i;k<3;++k)
            {
                const int n=pairIndex(i,k);const double h=c[i]*c[k]-(i==k ? 1.0/3 : 0);
                p[n]+=f*h;p2[n]+=f2*h;
            }
            a3[0]+=f*(c[0]*c[0]-1.0/3)*c[1];a3[1]+=f*(c[0]*c[0]-1.0/3)*c[2];
            a3[2]+=f*(c[1]*c[1]-1.0/3)*c[0];a3[3]+=f*(c[1]*c[1]-1.0/3)*c[2];
            a3[4]+=f*(c[2]*c[2]-1.0/3)*c[0];a3[5]+=f*(c[2]*c[2]-1.0/3)*c[1];
            a3[6]+=f*c[0]*c[1]*c[2];
            // The diagonal third-order Hermite polynomials vanish exactly.
            for(int i=0;i<3;++i) close(c[i]*c[i]*c[i]-c[i],0,0);
        });
        close(rho,s.rho);close(rho2,s.rho);
        for(int i=0;i<3;++i) {close(j[i],s.rho*s.v[i]);close(j2[i],j[i]);}
        for(int i=0;i<3;++i) for(int k=i;k<3;++k)
        {const int n=pairIndex(i,k);close(p[n],s.rho*(s.v[i]*s.v[k]+s.p[n]));close(p2[n],p[n]);}
        const int idx[7][3]={{0,0,1},{0,0,2},{1,1,0},{1,1,2},{2,2,0},{2,2,1},{0,1,2}};
        for(int n=0;n<7;++n) close(a3[n],s.rho*rrThird(s,a,idx[n][0],idx[n][1],idx[n][2]));
        const real_t ws=1.3,wb=1.1;auto post=collideHydro(s,a,ws,wb);
        const double tr=(s.p[0]+s.p[1]+s.p[2])/3.0;
        double u[3];for(int i=0;i<3;++i) {u[i]=s.v[i]+a[i]/2;close(post.v[i]-s.v[i],a[i]);}
        // Independently evaluate the MRT second moment and Guo source at u=q+a/2.
        // Direct tensor form: collision on (A2-u*u), source with matching shear/bulk rates.
        double ntrace=0,sourceTrace=0;
        for(int i=0;i<3;++i) {ntrace+=s.v[i]*s.v[i]+s.p[i]-u[i]*u[i];sourceTrace+=2*u[i]*a[i];}
        ntrace/=3;sourceTrace/=3;
        for(int i=0;i<3;++i) for(int k=i;k<3;++k)
        {
            const int n=pairIndex(i,k);const double nt=i==k ? ntrace : 0,st=i==k ? sourceTrace : 0;
            const double neq=s.v[i]*s.v[k]+s.p[n]-u[i]*u[k];
            const double src=u[i]*a[k]+u[k]*a[i];
            const double expected=u[i]*u[k]+(1-ws)*(neq-nt)+(1-wb)*nt+(1-ws/2)*(src-st)+(1-wb/2)*st;
            close(post.v[i]*post.v[k]+post.p[n],expected);
        }
        // Independent third-order collision/source balance at the physical velocity.
        const auto postSingle=collideHydro(s,a,ws,ws);
        for(const auto &ijk:idx)
        {
            const int i=ijk[0],j=ijk[1],k=ijk[2];
            double rec=0,source=0;
            const int q[3]={i,j,k};
            for(int d=0;d<3;++d)
            {
                const int v=q[d],r=q[(d+1)%3],t=q[(d+2)%3];
                rec+=u[v]*(s.p[pairIndex(r,t)]+double(a[r])*a[t]/4);
                source+=a[v]*u[r]*u[t];
            }
            const double eq=u[i]*u[j]*u[k],preThird=eq+rec-source/2;
            const double expected=(1-ws)*preThird+ws*eq+(1-ws/2)*source;
            close(rrThird(postSingle,a,i,j,k),expected,5.e-10);
        }
        // RR2 and RR3 have the same first two moments even away from equilibrium.
    }
    HydroState rest{1,{0,0,0},{0.001,-0.001,0,0.002,0,0}};const real_t zero[3]={};
    constexpr_for<0,27>([&](auto q){close(hydroPopulation<q,2>(rest,zero),hydroPopulation<q,3>(rest,zero),0);});
}

__global__ void manufacture(real_t *m)
{
    const int z=threadIdx.x;if(z>=NZ) return;
    for(int y=0;y<NY;++y) for(int x=0;x<NX;++x)
    {
        const int idx=global3(x,y,z);
        for(int f=0;f<NUM_MOMENTS;++f) m[midx(idx,f)]=0;
        m[midx(idx,RHO)]=1;
        const double h=NZ-1,zn=z/h;
        // Exact prescribed Dirichlet data plus a decaying diffusion eigenmode.
        const double ux=0.01*sin(3.141592653589793*zn);
        m[midx(idx,UX)]=ux;
        m[midx(idx,PXZ)]=-(TAU_SHEAR-1)*0.01*3.141592653589793*cos(3.141592653589793*zn)/(3*h);
        m[midx(idx,TEMP)]=1-zn+0.05*sin(3.141592653589793*zn);
        m[midx(idx,TUX)]=m[midx(idx,TEMP)]*ux;
        const double grad=(-1+0.05*3.141592653589793*cos(3.141592653589793*zn))/h;
        m[midx(idx,TUZ)]=-(TAU_THERMAL-1)*grad/3;
    }
}
static void nee()
{
    std::vector<real_t> host(size_t(CELLS)*NUM_MOMENTS,0);
    real_t *device;outputCheckCuda(cudaMalloc(&device,host.size()*sizeof(real_t)),"NEE test allocation");
    for(natural_t z=1;z<NZ-1;++z) for(natural_t xy=0;xy<STRIDE;++xy)
    {
        const natural_t n=xy+z*STRIDE;host[midx(n,RHO)]=1.03;host[midx(n,TEMP)]=0.75;
        const real_t u[3]={0.01,-0.02,0.015};real_t a[3];acceleration(0.75,true,a);
        for(int d=0;d<3;++d) {host[midx(n,UX+d)]=u[d]+a[d]/2;host[midx(n,TUX+d)]=0.75*u[d]+0.001*(d+1);}
        for(int d=0;d<6;++d) host[midx(n,PXX+d)]=0.0001*(d+1);
    }
    outputCheckCuda(cudaMemcpy(device,host.data(),host.size()*sizeof(real_t),cudaMemcpyHostToDevice),"NEE test upload");
    applyNEEPlates<<<dim3((NX+31)/32,(NY+3)/4,2),dim3(32,4)>>>(device);
    outputCheckCuda(cudaMemcpy(host.data(),device,host.size()*sizeof(real_t),cudaMemcpyDeviceToHost),"NEE test read");
    real_t a[3];acceleration(0.75,true,a);
    for(natural_t z:{0u,NZ-1})
    {
        const natural_t n=z*STRIDE;
        close(host[midx(n,TEMP)],z==0 ? 1 : 0,0);close(host[midx(n,RHO)],1.03);
        HydroState wall{host[midx(n,RHO)],{}, {}};
        real_t jt[3];
        for(int d=0;d<3;++d)
        {close(host[midx(n,UX+d)],0,0);close(host[midx(n,TUX+d)],0.001*(d+1));jt[d]=host[midx(n,TUX+d)];}
        for(int d=0;d<6;++d)
        {
            const int i=d<3 ? d : d==3 || d==4 ? 0 : 1,j=d<3 ? d : d==3 ? 1 : 2;
            close(host[midx(n,PXX+d)],0.0001*(d+1)+a[i]*a[j]/4,1.e-10);
            wall.p[d]=host[midx(n,PXX+d)];
        }
        double rho=0,t=0,j[3]={};const real_t zero[3]={};
        constexpr_for<0,27>([&](auto q){
            const double f=hydroPopulation<q>(wall,zero);rho+=f;
            j[0]+=f*D3Q27::cx<q>();j[1]+=f*D3Q27::cy<q>();j[2]+=f*D3Q27::cz<q>();
            t+=thermalPopulation<q>(host[midx(n,TEMP)],jt);
        });
        close(rho,1.03);close(t,z==0 ? 1 : 0);
        for(int d=0;d<3;++d)close(j[d],0);
    }
    cudaFree(device);std::cout<<"NEE population moments, force removal, exact wall velocity/temperature: PASS\n";
}
__global__ void gradientField(real_t *m)
{
    const int z=threadIdx.x;if(z>=NZ) return;
    for(int y=0;y<NY;++y) for(int x=0;x<NX;++x)
    {
        const int n=global3(x,y,z);
        const real_t t=1-real_t(z)/(NZ-1);real_t a[3];acceleration(t,z>0 && z<NZ-1,a);
        m[midx(n,TEMP)]=t;m[midx(n,RHO)]=1;
        m[midx(n,UX)]=0.01*sin(2*3.141592653589793*y/NY)*sin(3.141592653589793*z/(NZ-1));
        m[midx(n,UY)]=0;m[midx(n,UZ)]=a[2]/2;
    }
}
static void gradientDiagnostics()
{
    real_t *m;outputCheckCuda(cudaMalloc(&m,size_t(CELLS)*NUM_MOMENTS*sizeof(real_t)),"gradient allocation");
    gradientField<<<1,NZ>>>(m);
    rbc::Statistics s;
    rbc::reduceTiles<<<dim3(rbc::TILES,NZ),rbc::THREADS>>>(m,s.partial);
    rbc::reducePlanes<<<rbc::PROFILE_SIZE,rbc::THREADS>>>(s.partial,s.deviceProfile);
    outputCheckCuda(cudaMemcpy(s.profile.data(),s.deviceProfile,rbc::PROFILE_SIZE*sizeof(double),cudaMemcpyDeviceToHost),"gradient profiles");
    const double kz=3.141592653589793/(NZ-1),ky=2*3.141592653589793/NY;
    for(unsigned z=0;z<NZ;++z)
    {
        double dz=std::cos(kz*z)*std::sin(kz);
        if(z==0) dz=(4*std::sin(kz)-std::sin(2*kz))/2;
        if(z==NZ-1) dz=(3*std::sin(kz*z)-4*std::sin(kz*(z-1))+std::sin(kz*(z-2)))/2;
        const double expected=VISCOSITY*0.0001/2*(std::pow(std::sin(ky)*std::sin(kz*z),2)+dz*dz);
        close(s.profile[z*rbc::NS+rbc::EU],expected,std::max(1.e-14,expected*1.e-4));
        close(s.profile[z*rbc::NS+rbc::ET],double(THERMAL_DIFFUSIVITY)/((NZ-1)*(NZ-1)),1.e-8);
    }
    cudaFree(m);std::cout<<"GPU gradient dissipation and FP64 profile reductions: PASS\n";
}
static void diffusion()
{
    real_t *m,*b;const size_t bytes=size_t(CELLS)*NUM_MOMENTS*sizeof(real_t);
    outputCheckCuda(cudaMalloc(&m,bytes),"test allocation");outputCheckCuda(cudaMalloc(&b,bytes),"test allocation");
    manufacture<<<1,NZ>>>(m);
    const dim3 grid(GRID_X,GRID_Y,GRID_Z),block(BLOCK_NX,BLOCK_NY,BLOCK_NZ);
    const auto plates=dim3((NX+31)/32,(NY+3)/4,2);
    applyNEEPlates<<<plates,dim3(32,4)>>>(m);
    const int steps=std::lround(0.02*(NZ-1)*(NZ-1)/THERMAL_DIFFUSIVITY);
    for(int t=0;t<steps;++t)
    {
        streamCollide<<<grid,block>>>(m,b,t);applyNEEPlates<<<plates,dim3(32,4)>>>(b);std::swap(m,b);
    }
    std::vector<real_t> t(CELLS),u(CELLS),ux(CELLS);
    outputCheckCuda(cudaMemcpy(t.data(),m+size_t(TEMP)*CELLS,CELLS*sizeof(real_t),cudaMemcpyDeviceToHost),"test read");
    outputCheckCuda(cudaMemcpy(u.data(),m+size_t(UZ)*CELLS,CELLS*sizeof(real_t),cudaMemcpyDeviceToHost),"test read");
    outputCheckCuda(cudaMemcpy(ux.data(),m+size_t(UX)*CELLS,CELLS*sizeof(real_t),cudaMemcpyDeviceToHost),"test shear read");
    double error=0,errorU=0;
    for(unsigned z=0;z<NZ;++z)
    {
        const double zn=z/double(NZ-1),exact=1-zn+0.05*std::sin(3.141592653589793*zn)*
            std::exp(-THERMAL_DIFFUSIVITY*steps*3.141592653589793*3.141592653589793/((NZ-1)*(NZ-1)));
        error=std::max(error,std::abs(t[z*STRIDE]-exact));
        const double exactU=0.01*std::sin(3.141592653589793*zn)*std::exp(-VISCOSITY*steps*3.141592653589793*3.141592653589793/((NZ-1)*(NZ-1)));
        errorU=std::max(errorU,std::abs(ux[z*STRIDE]-exactU));
        close(u[z*STRIDE],0,2.e-6);
    }
    close(t[0],1,0);close(t[(NZ-1)*STRIDE],0,0);
    const double wallFlux=(3*t[0]-4*t[STRIDE]+t[2*STRIDE])/2*(NZ-1);
    const double exactFlux=1-0.05*3.141592653589793*std::exp(-THERMAL_DIFFUSIVITY*steps*3.141592653589793*3.141592653589793/((NZ-1)*(NZ-1)));
    std::cout<<"diffusion "<<NZ<<' '<<error<<' '<<std::abs(wallFlux-exactFlux)<<' '<<errorU<<'\n';
    cudaFree(m);cudaFree(b);
}
int main(int argc,char **)
{
    quadrature();std::cout<<"moment quadrature, force/MRT consistency, Hermite degeneracies, RR2/RR3 limits: PASS\n";
    nee();gradientDiagnostics();
    if(argc>1) diffusion();
}
