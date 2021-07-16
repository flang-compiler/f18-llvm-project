! RUN: bbc %s -o - | FileCheck %s

! CHECK: _QQmain
program test_adjust
  character(len=32) :: flang_str = '  0123flang  '
  character(len=32) :: l_str, r_str

  l_str = adjustl(flang_str)
  r_str = adjustr(flang_str)

  print *, '''', flang_str, ''''
  print *, '''', l_str, ''''
  print *, '''', r_str, ''''

end program test_adjust
