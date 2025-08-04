//> using scala "2.13.12"
//> using dep "org.chipsalliance::chisel:7.0.0-RC3"
//> using plugin "org.chipsalliance:::chisel-plugin:7.0.0-RC3"
//> using options "-unchecked" "-deprecation" "-language:reflectiveCalls" "-feature" "-Xcheckinit" "-Xfatal-warnings" "-Ywarn-dead-code" "-Ywarn-unused" "-Ymacro-annotations"

import chisel3._
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
import _root_.circt.stage.ChiselStage

class AddSimple extends Module {
  val a = IO(Input(UInt(8.W)))
  val b = IO(Input(UInt(4.W)))
  val c = IO(Input(UInt(2.W)))
  val out = IO(Output(UInt(10.W)))

  val ab = a + b
  val abc = ab + c
  out := abc
}

object Main extends App {
  val firrtl = ChiselStage.emitFIRRTLDialect(gen = new AddSimple,
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
    )
  import java.io._
  val pw = new PrintWriter(new File("AddSimple.mlir"))
  pw.write(firrtl)
  pw.close()
  ChiselStage.emitSystemVerilogFile(new AddSimple)
}
