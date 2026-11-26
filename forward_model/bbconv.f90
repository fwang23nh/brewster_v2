subroutine prism(obspec, modspec,resel,Fratio_int, nobs, nmod, obs_rows, mod_rows)

  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  implicit none


  !f2py intent(inout) obspec,modspec
  !f2py intent(in) resel
  !f2py intent(out) Fratio_int
  !f2py intent(hide) nobs, nmod, obs_rows, mod_rows
  
  integer, intent(in) :: nobs, nmod, obs_rows, mod_rows
  double precision,intent(inout) :: obspec(obs_rows,nobs),modspec(mod_rows,nmod)
  double precision,dimension(nobs),intent(out) :: Fratio_int
  double precision,allocatable,dimension(:):: delta1,delta2,delta
  double precision,allocatable,dimension(:):: gauss,wlmod,Fmod,wlobs
  double precision:: sigma,resel
  integer :: i, lo, hi, mid, first, last
  logical :: windowable
  double precision :: weight, flux, maxflux, omitted, lower_wl, upper_wl
  double precision, parameter :: sigma_limit = 10d0, relative_tol = 1d-12
  double precision, parameter :: tail_weight = exp(-0.5d0 * sigma_limit**2)

  if (nobs == 0) return
  if (nobs == 1) then
     ! Prism width is defined by neighboring observed wavelengths.
     Fratio_int(1) = ieee_value(0d0, ieee_quiet_nan)
     return
  endif

  allocate(delta1(nobs),delta2(nobs),delta(nobs),wlobs(nobs))
  allocate(gauss(nmod),wlmod(nmod),Fmod(nmod))

  
  Fmod = modspec(2,:)
  wlmod = modspec(1,:)
  wlobs = obspec(1,:)
  ! Binary searches require a finite ascending grid.
  windowable = nmod > 0 .and. all(ieee_is_finite(wlmod)) .and. all(ieee_is_finite(Fmod))
  if (windowable) then
     windowable = all(wlmod(2:) >= wlmod(:nmod-1))
     maxflux = maxval(abs(Fmod))
  endif

  delta1 = abs(wlobs - cshift(wlobs,+1))
  delta2 = abs(wlobs - cshift(wlobs,-1))
  delta = 0.5*(delta1 + delta2)

  
  delta(1) = delta(2)
  delta(nobs) = delta(nobs - 1)

  do i = 1, nobs
     sigma = delta(i) * resel / 2.355
     if (windowable .and. ieee_is_finite(sigma) .and. sigma > 0d0 .and. ieee_is_finite(wlobs(i))) then
        lower_wl = wlobs(i) - sigma_limit * sigma
        upper_wl = wlobs(i) + sigma_limit * sigma
        lo = 1
        hi = nmod + 1
        do while (lo < hi)
           mid = (lo + hi) / 2
           if (wlmod(mid) < lower_wl) then
              lo = mid + 1
           else
              hi = mid
           endif
        enddo
        first = lo
        hi = nmod + 1
        do while (lo < hi)
           mid = (lo + hi) / 2
           if (wlmod(mid) <= upper_wl) then
              lo = mid + 1
           else
              hi = mid
           endif
        enddo
        last = lo - 1
        if (last >= first .and. last-first+1 < nmod) then
           gauss(first:last) = exp(-(wlmod(first:last) - wlobs(i))**2 / (2*sigma**2))
           weight = sum(gauss(first:last))
           gauss(first:last) = gauss(first:last) / weight
           flux = sum(gauss(first:last)*Fmod(first:last))
           ! Bound the omitted Gaussian weight and the resulting flux error,
           ! including renormalization. Use the full grid if this bound is too large.
           omitted = (nmod - (last - first + 1)) * tail_weight
           if ((maxflux+abs(flux))*omitted <= relative_tol*abs(flux)*weight) then
              Fratio_int(i) = flux
              cycle
           endif
        endif
     endif
     gauss = exp(-(wlmod - wlobs(i))**2 / (2*sigma**2))
     gauss = gauss / sum(gauss)
     Fratio_int(i) = sum(gauss*Fmod)
  end do

  deallocate(delta1,delta2,delta)
  deallocate(gauss,wlmod)

end subroutine prism

subroutine convFWHM(obspec,modspec,fwhm,Fratio_int, nobs, nmod, obs_rows, mod_rows)

  implicit none


  !f2py intent(inout) modspec,obspec
  !f2py intent(in) fwhm
  !f2py intent(out) Fratio_int
  !f2py intent(hide) nobs, nmod, obs_rows, mod_rows
  
  integer, intent(in) :: nobs, nmod, obs_rows, mod_rows
  double precision,intent(inout) :: obspec(obs_rows,nobs),modspec(mod_rows,nmod)
  double precision,dimension(nobs),intent(out) :: Fratio_int
  double precision,allocatable,dimension(:):: wlobs,gauss,wlmod,Fmod
  integer :: i
  double precision :: fwhm, sigma

  allocate(wlobs(nobs))
  allocate(gauss(nmod),wlmod(nmod),Fmod(nmod))

  Fmod = modspec(2,:)
  wlmod = modspec(1,:)
  wlobs = obspec(1,:)

  do i = 1, nobs
     sigma = fwhm / 2.355

     gauss = exp(-(wlmod-wlobs(i))**2 / (2*sigma**2))
     
     gauss = gauss/ sum(gauss)
     
     Fratio_int(i)= sum(gauss*Fmod)

  end do

  deallocate(wlobs)
  deallocate(gauss,wlmod,Fmod)

end subroutine convFWHM



subroutine convR(obspec,modspec,R,Fratio_int, nobs, nmod, obs_rows, mod_rows)

  implicit none


  !f2py intent(inout) modspec,obspec
  !f2py intent(in) R
  !f2py intent(out) Fratio_int
  !f2py intent(hide) nobs, nmod, obs_rows, mod_rows
  
  integer, intent(in) :: nobs, nmod, obs_rows, mod_rows
  double precision,intent(inout) :: obspec(obs_rows,nobs),modspec(mod_rows,nmod)
  double precision,dimension(nobs),intent(out) :: Fratio_int
  double precision,allocatable,dimension(:):: wlobs,gauss,wlmod,Fmod
  integer :: i
  double precision:: R, sigma

  allocate(wlobs(nobs))
  allocate(gauss(nmod),wlmod(nmod),Fmod(nmod))
  
  Fmod = modspec(2,:)
  wlmod = modspec(1,:)
  wlobs = obspec(1,:)

  do i = 1, nobs
     !sigma is FWHM / 2.355
     sigma = (wlobs(i) / R) / 2.355
     gauss = exp(-(wlmod-wlobs(i))**2/(2*sigma**2))
     gauss = gauss/ sum(gauss)
     Fratio_int(i)= sum(gauss*Fmod)
  end do

  deallocate(wlobs)
  deallocate(gauss,wlmod,Fmod)



end subroutine convR
