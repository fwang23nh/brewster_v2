subroutine marv(temp,logg,R2D2,ingasname,molmass,logVMR,pcover,&
     cloudmap,cloud_opaname,cloudsize,clouddata,miewave,mierad, &
     cloudrad,cloudsig,cloudprof,&
     inlinetemps,inpress,inwavenum,inlinelist,cia,ciatemps,&
     use_disort,make_cl_pspec,make_oth_pspec,make_cf,do_bff,bff,outspec,&
     cl_phot_press,oth_phot_press,cfunc)

  use sizes
  use main
  

  !f2py integer, parameter :: nlinetemps
  !f2py intent(in) logg,R2D2,ingasname,cloud_opaname
  !f2py intent(inout) temp,logVMR,inpress
  !f2py intent(in) use_disort,make_cl_pspec,make_oth_pspec,make_cf,do_bff
  !f2py intent(inout) inlinetemps
  !f2py intent(inout) cloudrad,cloudsig,cloudprof,cloudsize
  !f2py intent(inout) cia, ciatemps
  !f2py intent(inout) inlinelist, inwavenum,bff,clouddata,miewave,mierad
  !f2py intent(inout) cloudmap,pcover,molmass
  !f2py intent(out) out_spec, cl_phot_press,oth_phot_press,cfunc

  real,intent(inout) :: cia(:,:,:)
  real,dimension(nciatemps) :: ciatemps
  real,intent(inout) :: inlinelist(:,:,:,:)
  double precision,intent(inout):: temp(:)
  integer,intent(inout):: cloudsize(:)
  real :: R2D2,logg
  double precision,intent(inout):: bff(:,:)
  real,intent(inout) :: pcover(:),molmass(:)
  ! cloudmap is npatch,ncloud
  integer,intent(inout) ::cloudmap(:,:)
  character(len=15),intent(in) :: ingasname(:)
  character(len=50),intent(in) ::cloud_opaname(:)
  character(len=15),dimension(:),allocatable:: gasname
  character(len=50),dimension(:),allocatable :: cloudname
  double precision,intent(inout) :: logVMR(:,:)
  ! clouddata: ncloud,miewave,mierad,3(qext,qscat,cos_qscar)
  double precision,intent(inout) :: clouddata(:,:,:,:)
  double precision,intent(inout) :: mierad(:), miewave(:)
  double precision,intent(inout) :: cloudrad(:,:)
  double precision,intent(inout) :: cloudsig(:,:)
  double precision,intent(inout) :: cloudprof(:,:)
  double precision,dimension(2,maxwave),intent(OUT):: outspec
  double precision,dimension(maxpatch,maxwave),intent(OUT):: cl_phot_press,oth_phot_press
  double precision,dimension(:,:),allocatable :: out_spec, clphotspec,othphotspec
  double precision,dimension(:,:,:),allocatable :: cf
  double precision,dimension(maxpatch,maxwave,maxlayers) :: cfunc
  double precision,intent(inout) :: inwavenum(:)
  real,intent(inout) :: inlinetemps(:)
  real,intent(inout) :: inpress(:)
  integer :: use_disort,make_oth_pspec,make_cl_pspec,do_bff,make_cf
  logical :: othphot,clphot,do_cf

  !open (1, file = 'log.txt', status ='old')
 !write(1,*) 'here marv 56'
  call initlayers(size(inpress))
  call initwave(size(inwavenum))
  call initgas(size(molmass))
  call initpatch(size(cloudmap(:,1)))
  call initcloud(size(cloudmap(1,:)))
  call inittemps(size(inlinetemps))

  allocate(out_spec(2,nwave))

 !write(1,*) 'here marv 66'

  clphot = make_cl_pspec
  othphot = make_oth_pspec
  do_cf = make_cf
  
  allocate(gasname(size(ingasname)))
  allocate(cloudname(ncloud))
  gasname = ingasname
  cloudname = cloud_opaname
  
  call forward(temp,logg,R2D2,gasname,molmass,logVMR,pcover,&
       cloudmap,cloudname,cloudsize,clouddata,miewave,mierad,&
       cloudrad,cloudsig,cloudprof,&
       inlinetemps,inpress,inwavenum,inlinelist,cia,ciatemps,use_disort,&
       clphot,othphot,do_cf,do_bff,bff,out_spec,clphotspec,othphotspec,cf)

  outspec = 0.d0
  outspec(1,1:nwave)  = out_spec(1,1:nwave)
  outspec(2,1:nwave) = out_spec(2,1:nwave)

  cl_phot_press  = 0.d0
  if (clphot) then
     do ipatch = 1, npatch
        cl_phot_press(ipatch,1:nwave) = clphotspec(ipatch,1:nwave)
     end do
  end if
  oth_phot_press  = 0.d0
  if (othphot) then
     do ipatch = 1, npatch
        oth_phot_press(ipatch,1:nwave) = othphotspec(ipatch,1:nwave)
     end do
  end if

  cfunc = 0.d0
  if (do_cf) then
     do ipatch = 1, npatch
        do ilayer = 1, nlayers
           cfunc(ipatch,1:nwave,ilayer)  = cf(ipatch,1:nwave,ilayer)
        end do
     end do
  end if

  deallocate(out_spec,clphotspec,othphotspec,cf)
  !close(1)
end subroutine marv


