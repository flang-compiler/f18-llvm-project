! RUN: bbc %s -o - | \
!      sed '/cmp[if] [^"]/{s|cmp[if] |&"|; s|,|"&|}' | \
!      tco | llc --relocation-model=pic --filetype=obj -o %t.o
! RUN: %CC %t.o -L%L -lFortranRuntime -lFortranDecimal -lstdc++ -lm -o %t.out
! RUN: %t.out | FileCheck %s

  function sinteger(n)
    integer sinteger
    nn = -88
    select case(n)
    case (:1)
      nn = 1
    case (2)
      nn = 2
    case default
      nn = 0
    case (3)
      nn = 3
    case (4:5+1-1)
      nn = 4
    case (6)
      nn = 6
    case (7,8:15,21:)
      nn = 7
    end select
    sinteger = nn
  end

  subroutine slogical(L)
    logical :: L
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    n6 = 0
    n7 = 0
    n8 = 0

    select case (L)
    end select

    select case (L)
      case (.false.)
        n2 = 1
    end select

    select case (L)
      case (.true.)
        n3 = 2
    end select

    select case (L)
      case default
        n4 = 3
    end select

    select case (L)
      case (.false.)
        n5 = 1
      case (.true.)
        n5 = 2
    end select

    select case (L)
      case (.false.)
        n6 = 1
      case default
        n6 = 3
    end select

    select case (L)
      case (.true.)
        n7 = 2
      case default
        n7 = 3
    end select

    select case (L)
      case (.false.)
        n8 = 1
      case (.true.)
        n8 = 2
      case default
        n8 = 3
    end select

    print*, n1, n2, n3, n4, n5, n6, n7, n8
  end

  program p
    integer sinteger, v(10)

    n = -10
    do j = 1, 4
      do k = 1, 10
        n = n + 1
        v(k) = sinteger(n)
      enddo
      ! CHECK: 1 1 1 1 1 1 1 1 1 1
      ! CHECK: 1 2 3 4 4 6 7 7 7 7
      ! CHECK: 7 7 7 7 7 0 0 0 0 0
      ! CHECK: 7 7 7 7 7 7 7 7 7 7
      print*, v
    enddo

    print*
    ! CHECK: 0 1 0 3 1 1 3 1
    call slogical(.false.)
    ! CHECK: 0 0 2 3 2 3 2 2
    call slogical(.true.)
  end
