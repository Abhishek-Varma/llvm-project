// RUN: mlir-opt -split-input-file -transform-interpreter -cse %s | FileCheck %s

// Test vectorization of batch-less convolutions, which are linalg.generic ops
// with convolution semantics but without a batch dimension. These can arise
// from IREE's DispatchCreation which strips unit dimensions including N=1.

// Batch-less 1D convolution in NWC-like layout (WC layout after batch stripping)
// Input: WxC (8x4), Filter: KWxCxF (3x4x8), Output: WxF (6x8)
// This is equivalent to a batched conv with N=1 stripped.
func.func @conv1d_batchless_nwc(%input: tensor<8x4xf32>,
                                 %filter: tensor<3x4x8xf32>,
                                 %output: tensor<6x8xf32>) -> tensor<6x8xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0 + d2, d3)>,     // input: [w+kw, c]
      affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>,      // filter: [kw, c, f]
      affine_map<(d0, d1, d2, d3) -> (d0, d1)>           // output: [w, f]
    ],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<8x4xf32>, tensor<3x4x8xf32>)
    outs(%output : tensor<6x8xf32>) {
  ^bb0(%in: f32, %flt: f32, %out: f32):
    %mul = arith.mulf %in, %flt : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<6x8xf32>
  return %0 : tensor<6x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @conv1d_batchless_nwc
// Verify that we read 2D tensors and expand to 3D for computation
// CHECK:       vector.transfer_read {{.*}} : tensor<8x4xf32>
// CHECK:       vector.transfer_read {{.*}} : tensor<3x4x8xf32>
// CHECK:       vector.transfer_read {{.*}} : tensor<6x8xf32>
// Verify convolution computation with vector.contract
// CHECK:       vector.contract
// Verify that we write back to 2D tensor
// CHECK:       vector.transfer_write {{.*}} : {{.*}}, tensor<6x8xf32>

// -----

// Batch-less 1D convolution in NCW-like layout (CW layout after batch stripping)
// Input: CxW (4x8), Filter: FxCxKW (8x4x3), Output: FxW (8x6)
func.func @conv1d_batchless_ncw(%input: tensor<4x8xf32>,
                                 %filter: tensor<8x4x3xf32>,
                                 %output: tensor<8x6xf32>) -> tensor<8x6xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d2, d1 + d3)>,     // input: [c, w+kw]
      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,      // filter: [f, c, kw]
      affine_map<(d0, d1, d2, d3) -> (d0, d1)>           // output: [f, w]
    ],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<4x8xf32>, tensor<8x4x3xf32>)
    outs(%output : tensor<8x6xf32>) {
  ^bb0(%in: f32, %flt: f32, %out: f32):
    %mul = arith.mulf %in, %flt : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<8x6xf32>
  return %0 : tensor<8x6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @conv1d_batchless_ncw
// Verify that we read 2D tensors and expand to 3D for computation
// CHECK:       vector.transfer_read {{.*}} : tensor<4x8xf32>
// CHECK:       vector.transfer_read {{.*}} : tensor<8x4x3xf32>
// CHECK:       vector.transfer_read {{.*}} : tensor<8x6xf32>
// Verify convolution computation with vector.contract
// CHECK:       vector.contract
// Verify that we write back to 2D tensor
// CHECK:       vector.transfer_write {{.*}} : {{.*}}, tensor<8x6xf32>

// -----

// Batched 1D convolution in NWC layout for comparison
// Input: NxWxC (1x8x4), Filter: KWxCxF (3x4x8), Output: NxWxF (1x6x8)
func.func @conv1d_batched_nwc(%input: tensor<1x8x4xf32>,
                               %filter: tensor<3x4x8xf32>,
                               %output: tensor<1x6x8xf32>) -> tensor<1x6x8xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>,  // input: [n, w+kw, c]
      affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>,       // filter: [kw, c, f]
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>        // output: [n, w, f]
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x8x4xf32>, tensor<3x4x8xf32>)
    outs(%output : tensor<1x6x8xf32>) {
  ^bb0(%in: f32, %flt: f32, %out: f32):
    %mul = arith.mulf %in, %flt : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<1x6x8xf32>
  return %0 : tensor<1x6x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @conv1d_batched_nwc
// Verify that we read/write 3D tensors
// CHECK:       vector.transfer_read {{.*}} : tensor<1x8x4xf32>, vector<1x8x4xf32>
// CHECK:       vector.transfer_read {{.*}} : tensor<3x4x8xf32>, vector<3x4x8xf32>
// CHECK:       vector.transfer_read {{.*}} : tensor<1x6x8xf32>, vector<1x6x8xf32>
// Verify convolution computation
// CHECK:       vector.contract
// CHECK:       vector.transfer_write {{.*}} : vector<1x6x8xf32>, tensor<1x6x8xf32>
