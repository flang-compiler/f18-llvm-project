;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 -mattr=+m %s -o - | FileCheck --check-prefix=RV32M %s
; RUN: llc -mtriple=riscv32 -mattr=+a %s -o - | FileCheck --check-prefix=RV32A %s
; RUN: llc -mtriple=riscv32 -mattr=+f %s -o - | FileCheck --check-prefix=RV32F %s
; RUN: llc -mtriple=riscv32 -mattr=+d %s -o - | FileCheck --check-prefix=RV32D %s
; RUN: llc -mtriple=riscv32 -mattr=+c %s -o - | FileCheck --check-prefix=RV32C %s
; RUN: llc -mtriple=riscv32 -mattr=+zfhmin %s -o - | FileCheck --check-prefix=RV32ZFHMIN %s
; RUN: llc -mtriple=riscv32 -mattr=+zfh %s -o - | FileCheck --check-prefix=RV32ZFH %s
; RUN: llc -mtriple=riscv32 -mattr=+zba %s -o - | FileCheck --check-prefix=RV32ZBA %s
; RUN: llc -mtriple=riscv32 -mattr=+zbb %s -o - | FileCheck --check-prefix=RV32ZBB %s
; RUN: llc -mtriple=riscv32 -mattr=+zbc %s -o - | FileCheck --check-prefix=RV32ZBC %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbe %s -o - | FileCheck --check-prefix=RV32ZBE %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbf %s -o - | FileCheck --check-prefix=RV32ZBF %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbm %s -o - | FileCheck --check-prefix=RV32ZBM %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbp %s -o - | FileCheck --check-prefix=RV32ZBP %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbr %s -o - | FileCheck --check-prefix=RV32ZBR %s
; RUN: llc -mtriple=riscv32 -mattr=+zbs %s -o - | FileCheck --check-prefix=RV32ZBS %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbt %s -o - | FileCheck --check-prefix=RV32ZBT %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-v %s -o - | FileCheck --check-prefix=RV32V %s
; RUN: llc -mtriple=riscv32 -mattr=+zbb,+zfh,+experimental-v,+f %s -o - | FileCheck --check-prefix=RV32COMBINED %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb %s -o - | FileCheck --check-prefix=RV32ZBKB %s
; RUN: llc -mtriple=riscv64 -mattr=+m %s -o - | FileCheck --check-prefix=RV64M %s
; RUN: llc -mtriple=riscv64 -mattr=+a %s -o - | FileCheck --check-prefix=RV64A %s
; RUN: llc -mtriple=riscv64 -mattr=+f %s -o - | FileCheck --check-prefix=RV64F %s
; RUN: llc -mtriple=riscv64 -mattr=+d %s -o - | FileCheck --check-prefix=RV64D %s
; RUN: llc -mtriple=riscv64 -mattr=+c %s -o - | FileCheck --check-prefix=RV64C %s
; RUN: llc -mtriple=riscv64 -mattr=+zfhmin %s -o - | FileCheck --check-prefix=RV64ZFHMIN %s
; RUN: llc -mtriple=riscv64 -mattr=+zfh %s -o - | FileCheck --check-prefix=RV64ZFH %s
; RUN: llc -mtriple=riscv64 -mattr=+zba %s -o - | FileCheck --check-prefix=RV64ZBA %s
; RUN: llc -mtriple=riscv64 -mattr=+zbb %s -o - | FileCheck --check-prefix=RV64ZBB %s
; RUN: llc -mtriple=riscv64 -mattr=+zbc %s -o - | FileCheck --check-prefix=RV64ZBC %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbe %s -o - | FileCheck --check-prefix=RV64ZBE %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbf %s -o - | FileCheck --check-prefix=RV64ZBF %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbm %s -o - | FileCheck --check-prefix=RV64ZBM %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbp %s -o - | FileCheck --check-prefix=RV64ZBP %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbr %s -o - | FileCheck --check-prefix=RV64ZBR %s
; RUN: llc -mtriple=riscv64 -mattr=+zbs %s -o - | FileCheck --check-prefix=RV64ZBS %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbt %s -o - | FileCheck --check-prefix=RV64ZBT %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v %s -o - | FileCheck --check-prefix=RV64V %s
; RUN: llc -mtriple=riscv64 -mattr=+zbb,+zfh,+experimental-v,+f %s -o - | FileCheck --check-prefix=RV64COMBINED %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb %s -o - | FileCheck --check-prefix=RV64ZBKB %s


; RV32M: .attribute 5, "rv32i2p0_m2p0"
; RV32A: .attribute 5, "rv32i2p0_a2p0"
; RV32F: .attribute 5, "rv32i2p0_f2p0"
; RV32D: .attribute 5, "rv32i2p0_f2p0_d2p0"
; RV32C: .attribute 5, "rv32i2p0_c2p0"
; RV32ZFHMIN: .attribute 5, "rv32i2p0_f2p0_zfhmin1p0"
; RV32ZFH: .attribute 5, "rv32i2p0_f2p0_zfh1p0_zfhmin1p0"
; RV32ZBA: .attribute 5, "rv32i2p0_zba1p0"
; RV32ZBB: .attribute 5, "rv32i2p0_zbb1p0"
; RV32ZBC: .attribute 5, "rv32i2p0_zbc1p0"
; RV32ZBE: .attribute 5, "rv32i2p0_zbe0p93"
; RV32ZBF: .attribute 5, "rv32i2p0_zbf0p93"
; RV32ZBM: .attribute 5, "rv32i2p0_zbm0p93"
; RV32ZBP: .attribute 5, "rv32i2p0_zbp0p93"
; RV32ZBR: .attribute 5, "rv32i2p0_zbr0p93"
; RV32ZBS: .attribute 5, "rv32i2p0_zbs1p0"
; RV32ZBT: .attribute 5, "rv32i2p0_zbt0p93"
; RV32V: .attribute 5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV32COMBINED: .attribute 5, "rv32i2p0_f2p0_d2p0_v1p0_zfh1p0_zfhmin1p0_zbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV32ZBKB: .attribute 5, "rv32i2p0_zbkb1p0"

; RV64M: .attribute 5, "rv64i2p0_m2p0"
; RV64A: .attribute 5, "rv64i2p0_a2p0"
; RV64F: .attribute 5, "rv64i2p0_f2p0"
; RV64D: .attribute 5, "rv64i2p0_f2p0_d2p0"
; RV64C: .attribute 5, "rv64i2p0_c2p0"
; RV64ZFHMIN: .attribute 5, "rv64i2p0_f2p0_zfhmin1p0"
; RV64ZFH: .attribute 5, "rv64i2p0_f2p0_zfh1p0_zfhmin1p0"
; RV64ZBA: .attribute 5, "rv64i2p0_zba1p0"
; RV64ZBB: .attribute 5, "rv64i2p0_zbb1p0"
; RV64ZBC: .attribute 5, "rv64i2p0_zbc1p0"
; RV64ZBE: .attribute 5, "rv64i2p0_zbe0p93"
; RV64ZBF: .attribute 5, "rv64i2p0_zbf0p93"
; RV64ZBM: .attribute 5, "rv64i2p0_zbm0p93"
; RV64ZBP: .attribute 5, "rv64i2p0_zbp0p93"
; RV64ZBR: .attribute 5, "rv64i2p0_zbr0p93"
; RV64ZBS: .attribute 5, "rv64i2p0_zbs1p0"
; RV64ZBT: .attribute 5, "rv64i2p0_zbt0p93"
; RV64V: .attribute 5, "rv64i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV64COMBINED: .attribute 5, "rv64i2p0_f2p0_d2p0_v1p0_zfh1p0_zfhmin1p0_zbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV64ZBKB: .attribute 5, "rv64i2p0_zbkb1p0"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
