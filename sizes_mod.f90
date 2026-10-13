module sizes

  implicit none


  ! number of condensate species
  integer, protected :: nclouds

  ! number of gases for mixing
  integer, protected :: ngas

  ! number of patches
  integer,protected :: npatch
  
  ! declares and values size parameters for arrays 
  
  ! number of pressure layers
  integer, protected :: nlayers
  
  ! number of temperatures in line list grid
  
  integer, protected :: nlinetemps
 
  
  ! number of temperatures in CIA tables
  integer,protected :: nciatemps

  
  ! number of wavelength/number bins in full range
  integer,protected :: nwave

  save

  
contains
  subroutine initwave(val)
    integer :: val
    nwave = val
  end subroutine initwave

  subroutine initgas(gasval)
    integer :: gasval
    ngas = gasval
  end subroutine initgas

  subroutine initpatch(pval)
    integer :: pval
    npatch =  pval
  end subroutine initpatch

  subroutine initcloud(pval)
    integer :: pval
    nclouds =  pval
  end subroutine initcloud

  subroutine initlayers(pval)
    integer :: pval
    nlayers =  pval
  end subroutine initlayers

  subroutine inittemps(pval)
    integer :: pval
    nlinetemps =  pval
  end subroutine inittemps

  subroutine initcia(pval)
    integer :: pval
    nciatemps = pval
  end subroutine initcia

end module sizes
