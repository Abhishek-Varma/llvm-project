; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc -mtriple=armv7 %s -o - | FileCheck %s

declare double @fn()

define void @test(ptr %p, ptr %res) nounwind {
; CHECK-LABEL: test:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    push {r4, lr}
; CHECK-NEXT:    vpush {d8}
; CHECK-NEXT:    vldr d8, [r0]
; CHECK-NEXT:    mov r4, r1
; CHECK-NEXT:    vcmp.f64 d8, #0
; CHECK-NEXT:    vmrs APSR_nzcv, fpscr
; CHECK-NEXT:    vneg.f64 d16, d8
; CHECK-NEXT:    vmov.f64 d17, d8
; CHECK-NEXT:    vmovne.f64 d17, d16
; CHECK-NEXT:    vstr d17, [r1]
; CHECK-NEXT:    bl fn
; CHECK-NEXT:    vcmp.f64 d8, #0
; CHECK-NEXT:    vmrs APSR_nzcv, fpscr
; CHECK-NEXT:    vmov d16, r0, r1
; CHECK-NEXT:    eor r1, r1, #-2147483648
; CHECK-NEXT:    vmov d17, r0, r1
; CHECK-NEXT:    vmovne.f64 d16, d17
; CHECK-NEXT:    vstr d16, [r4]
; CHECK-NEXT:    vpop {d8}
; CHECK-NEXT:    pop {r4, pc}
entry:
  %x = load double, ptr %p
  %cmp = fcmp une double %x, 0.000000e+00
  %nx = fneg double %x
  %sx = select i1 %cmp, double %nx, double %x
  store double %sx, ptr %res
  %y = call double @fn()
  %ny = fneg double %y
  %sy = select i1 %cmp, double %ny, double %y
  store double %sy, ptr %res
  ret void
}
