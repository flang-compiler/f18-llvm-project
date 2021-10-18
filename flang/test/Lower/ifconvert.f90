! RUN: bbc -fdebug-dump-pre-fir %s |& FileCheck %s

! Note: PFT dump output is fairly stable, including node indexes and
!       annotations, so all output is CHECKed.

  ! CHECK: 1 Program <anonymous>
  ! CHECK:   1 PrintStmt: print*
  print*

  ! CHECK:   <<DoConstruct>> -> 8
  ! CHECK:     2 NonLabelDoStmt -> 7: do i = 1, 5
  ! CHECK:     <<IfConstruct>> -> 7
  ! CHECK:       3 ^IfStmt [negate] -> 7: if(i <= 1 .or. i >= 5) cycle
  ! CHECK:       6 ^PrintStmt: print*, i
  ! CHECK:       5 EndIfStmt
  ! CHECK:     <<End IfConstruct>>
  ! CHECK:     7 EndDoStmt -> 2: end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, 5
     if (i <= 1 .or. i >= 5) cycle
     print*, i
  end do

  ! CHECK:   8 PrintStmt: print*
  print*

  ! CHECK:   <<DoConstruct>> -> 15
  ! CHECK:     9 NonLabelDoStmt -> 14: do i = 1, 5
  ! CHECK:     <<IfConstruct>> -> 14
  ! CHECK:       10 ^IfStmt [negate] -> 14: if(i <= 1 .or. i >= 5) cycle
  ! CHECK:       13 ^PrintStmt: print*, i
  ! CHECK:       12 EndIfStmt
  ! CHECK:     <<End IfConstruct>>
  ! CHECK:     14 EndDoStmt -> 9: 2 end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, 5
     if (i <= 1 .or. i >= 5) cycle
     print*, i
2 end do

  ! CHECK:   15 PrintStmt: print*
  print*

  ! CHECK:   <<DoConstruct>> -> 25
  ! CHECK:     16 NonLabelDoStmt -> 24: abc: do i = 1, 5
  ! CHECK:     <<IfConstruct>> -> 24
  ! CHECK:       17 ^IfStmt [negate] -> 24: if(i <= 1 .or. i >= 5) cycle abc
  ! CHECK:       <<IfConstruct>> -> 24
  ! CHECK:         20 ^IfStmt [negate] -> 24: if(i == 3) goto 3
  ! CHECK:         23 ^PrintStmt: print*, i
  ! CHECK:         22 EndIfStmt
  ! CHECK:       <<End IfConstruct>>
  ! CHECK:       19 EndIfStmt
  ! CHECK:     <<End IfConstruct>>
  ! CHECK:     24 EndDoStmt -> 16: 3 end do abc
  ! CHECK:   <<End DoConstruct>>
  abc: do i = 1, 5
     if (i <= 1 .or. i >= 5) cycle abc
     if (i == 3) goto 3
     print*, i
3 end do abc

  ! CHECK:   25 PrintStmt: print*
  print*

  ! CHECK:   <<DoConstruct>> -> 35
  ! CHECK:     26 NonLabelDoStmt -> 34: do i = 1, 5
  ! CHECK:     <<IfConstruct>> -> 34
  ! CHECK:       27 ^IfStmt [negate] -> 34: if(i == 3) goto 4
  ! CHECK:       <<IfConstruct>> -> 34
  ! CHECK:         30 ^IfStmt [negate] -> 34: if(i <= 1 .or. i >= 5) cycle
  ! CHECK:         33 ^PrintStmt: print*, i
  ! CHECK:         32 EndIfStmt
  ! CHECK:       <<End IfConstruct>>
  ! CHECK:       29 EndIfStmt
  ! CHECK:     <<End IfConstruct>>
  ! CHECK:     34 EndDoStmt -> 26: 4 end do
  ! CHECK:   <<End DoConstruct>>
  ! CHECK:   35 EndProgramStmt: end
  ! CHECK: End Program <anonymous>
  do i = 1, 5
     if (i == 3) goto 4
     if (i <= 1 .or. i >= 5) cycle
     print*, i
4 end do
end
