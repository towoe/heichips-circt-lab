module {
  hw.module @addition(in %clk : i1, in %a : i3, in %b : i2, in %c : i2, out q : i5) {
    %false = hw.constant false
    %c0_i2 = hw.constant 0 : i2
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %false, %a : i1, i3
    %1 = comb.concat %c0_i2, %b : i2, i2
    %2 = comb.add %0, %1 : i4
    %3 = comb.concat %false, %2 : i1, i4
    %4 = comb.concat %c0_i3, %c : i3, i2
    %5 = comb.add %3, %4 : i5
    hw.output %5 : i5
  }
}
