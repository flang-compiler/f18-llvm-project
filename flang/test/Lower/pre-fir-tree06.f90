! RUN: bbc -I %moddir -pft-test -fopenmp -o %t %s | FileCheck %s

! CHECK: 1 Program im
! CHECK-NEXT <<DoConstruct!>> -> 5
! CHECK: 5 <<OpenMPConstruct>>
! CHECK-NEXT: 6 AssignmentStmt: inrdone = .true.
! CHECK-NEXT:  <<End OpenMPConstruct>>
! CHECK-NEXT  7 EndProgramStmt: end program im


program im
  LOGICAL :: inrdone
  INTEGER :: i
  do i = 1, 100
  goto 2
2 inrdone = .false.
  end do

  !$OMP MASTER
  inrdone = .TRUE.
  !$OMP END MASTER
end program im
