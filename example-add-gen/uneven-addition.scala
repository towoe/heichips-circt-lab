//> using scala "2.13.12"
//> using dep "org.chipsalliance::chisel:7.0.0-RC3"
//> using plugin "org.chipsalliance:::chisel-plugin:7.0.0-RC3"
//> using options "-unchecked" "-deprecation" "-language:reflectiveCalls" "-feature" "-Xcheckinit" "-Xfatal-warnings" "-Ywarn-dead-code" "-Ywarn-unused" "-Ymacro-annotations"

import chisel3._
import chisel3.util.log2Ceil
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
import _root_.circt.stage.ChiselStage


class UnevenAddition(val n: Int) extends Module {
  val io = IO(new Bundle {
    val a = Input(Vec(n, UInt(8.W)))  // e.g. weights
    val b = Input(Vec(n, UInt(4.W)))  // e.g. activations
    val c = Input(Vec(n, UInt(2.W)))  // e.g. offset
    val out = Output(UInt((12 + log2Ceil(n)).W)) // 8+4 bits + possible carry
  })

  // TODO: Add inputs together with `+&`
  val products =

  // TODO: Add intermediate results in a tree structure to one final value
  def reduceTree(xs: Seq[UInt]): UInt = xs match {
  }

  io.out := reduceTree(products)
}


object Main extends App {
  val firrtl = ChiselStage.emitFIRRTLDialect(gen = new UnevenAddition(64),
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
    )
  import java.io._
  val pw = new PrintWriter(new File("UnevenAddition.mlir"))
  pw.write(firrtl)
  pw.close()
  ChiselStage.emitSystemVerilogFile(new UnevenAddition(64))
}
