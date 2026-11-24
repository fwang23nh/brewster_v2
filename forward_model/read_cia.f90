subroutine read_cia(filename,ncwave,wavenum,outcia,ciatemps)

  use sizes
  
  implicit none

  !f2py integer, parameter :: nciatemps
  !f2py integer :: nwave
  
  !f2py intent(in) :: ncwave
  !f2py intent(in) filename
  !f2py intent(inout) wavenum
  !f2py intent(inout) outcia
  !f2py intent(inout) ciatemps
  
  ! Preserve the full caller-supplied path, including absolute paths.
  character(len=*),intent(in):: filename
  real, allocatable, dimension(:,:,:) :: oldcia
  real :: outcia(:,:,:)
  real :: ciatemps(:)
  integer:: iciatemp,icwaven,iwave,oldw1,oldw2,idum1,idum2,i,ncwave
  double precision, intent(inout) :: wavenum(:)
  real, allocatable, dimension(:) :: ciawaven,wdiff
  real:: intfact, fdum1

  call initwave(size(wavenum))
  call initcia(size(ciatemps))

  allocate(oldcia(4,nciatemps,ncwave))
  allocate(ciawaven(ncwave),wdiff(ncwave))
  
  outcia = 0.0
  open(15,file=filename,status="old")
  do i=1, 3
     read(15,*)
  end do
  
  read(15,*) idum1, idum2

  
  if (idum1 .ne. ncwave .or. idum2 .ne. nciatemps) then
     write(*,*) " Problem with low-res CIA table : ", trim(filename)
     stop
  end if

 
  do iciatemp = 1, nciatemps
     read(15,*) ciatemps(iciatemp)
     do icwaven = 1, ncwave
        read(15,*) ciawaven(icwaven), oldcia(1,iciatemp,icwaven), &
             oldcia(2,iciatemp,icwaven), &
             oldcia(3,iciatemp,icwaven), oldcia(4,iciatemp,icwaven), fdum1

     end do
  end do
  close(15)


  ! Now we need to resample to higher resolution
  do iwave= 1 , nwave
     
     wdiff = abs(ciawaven - wavenum(iwave))
     
     oldw1 = minloc(wdiff,1)
     
     if (ciawaven(oldw1) .lt. wavenum(iwave)) then
        oldw2 = oldw1 + 1
     else
        oldw2 = oldw1
        oldw1 = oldw2 - 1
     end if
     
     intfact = (log10(wavenum(iwave)) - log10(ciawaven(oldw1))) / &
          (log10(ciawaven(oldw2)) - log10(ciawaven(oldw1)))


     outcia(1,:,iwave) = ((oldcia(1,:,oldw2) - oldcia(1,:,oldw1)) * intfact) &
          + oldcia(1,:,oldw1)
     outcia(2,:,iwave) = ((oldcia(2,:,oldw2) - oldcia(2,:,oldw1)) * intfact) &
          + oldcia(2,:,oldw1)
     outcia(3,:,iwave) = ((oldcia(3,:,oldw2) - oldcia(3,:,oldw1)) * intfact) &
          + oldcia(3,:,oldw1)
     outcia(4,:,iwave) = ((oldcia(4,:,oldw2) - oldcia(4,:,oldw1)) * intfact) &
          + oldcia(4,:,oldw1)

  end do  ! wave do

  deallocate(oldcia)
  deallocate(ciawaven,wdiff)
  
end subroutine read_cia
