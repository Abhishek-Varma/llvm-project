// RUN: mlir-opt --transform-interpreter --cse --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @fuse_tileable_consumer_scf_for(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1 = scf.for %arg3 = %c0 to %c64 step %c4 iter_args(%arg4 = %arg2) -> (tensor<64xf32>) {
      %extracted_slice = tensor.extract_slice %arg4[%arg3] [32] [1] : tensor<64xf32> to tensor<32xf32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice : tensor<32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %13 = arith.mulf %in, %in_16 : f32
          %14 = arith.addf %out, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<32xf32>
      %4 = tensor.insert_slice %3 into %arg4[%arg3] [32] [1] : tensor<32xf32> into tensor<64xf32>
      scf.yield %4 : tensor<64xf32>
    }
    %in_operand_2 = tensor.empty() : tensor<64xf32>
    %out_operand_3 = tensor.empty() : tensor<64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %in_operand_2 : tensor<64xf32>, tensor<64xf32>) outs(%out_operand_3 : tensor<64xf32>) -> tensor<64xf32>
    return %2 : tensor<64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %yield = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %yield use_for true
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @fuse_tileable_consumer_scf_for(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64xf32>)
//      CHECK:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   %0 = tensor.empty() : tensor<64xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]]
// CHECK-SAME:      iter_args(%[[MAT_OUT_ARG:.*]] = %[[ARG2]], %[[ELEM_OUT_ARG:.*]] = %0)
// CHECK-SAME:   {
//      CHECK:      %[[MAT_OUT_SLICE:.*]] = tensor.extract_slice %[[MAT_OUT_ARG]]
//      CHECK:      %[[MAT_OUT:.*]] = linalg.generic
// CHECK-SAME:              outs(%[[MAT_OUT_SLICE]] : tensor<32xf32>)
//      CHECK:      %[[SLICE_OPERAND2:.*]] = tensor.extract_slice %0[%[[IV]]] [32] [1]
//      CHECK:      %[[SLICE_OUT:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      %[[ELEM_OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK-SAME:              ins(%[[MAT_OUT]], %[[SLICE_OPERAND2]] :
// CHECK-SAME:              outs(%[[SLICE_OUT]] :
//      CHECK:      %[[INSERT_ELEM:.*]] = tensor.insert_slice %[[ELEM_OUT]] into %[[ELEM_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      %[[INSERT_MAT:.*]] = tensor.insert_slice %[[MAT_OUT]] into %[[MAT_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      scf.yield %[[INSERT_MAT]], %[[INSERT_ELEM]] :
//      CHECK:   }
//      CHECK:   return %[[FINAL_RESULT]]#1 :

// -----

module {
  func.func @fuse_tileable_consumer_scf_forall(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %arg2) -> (tensor<64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg5[%arg3, %arg4] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
      %3 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.forall.in_parallel {
         tensor.parallel_insert_slice %3 into %arg5[%arg3, %arg4] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
      }
    }
    %in_operand_2 = tensor.empty() : tensor<64x64xf32>
    %out_operand_3 = tensor.empty() : tensor<64x64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %in_operand_2 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%out_operand_3 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %2 : tensor<64x64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %yield = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %yield use_for false
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @fuse_tileable_consumer_scf_forall(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64x64xf32>)
//      CHECK:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<64x64xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:2 = scf.forall
// CHECK-SAME:      shared_outs(%[[MAT_OUT_ARG:.*]] = %[[ARG2]], %[[ELEM_OUT_ARG:.*]] = %[[OUT_INIT]])
// CHECK-SAME:   {
//      CHECK:      %[[MAT_OUT_SLICE:.*]] = tensor.extract_slice %[[MAT_OUT_ARG]]
//      CHECK:      %[[MAT_OUT:.*]] = linalg.matmul
// CHECK-SAME:              outs(%[[MAT_OUT_SLICE]] :
//      CHECK:      %[[SLICE_OPERAND2:.*]] = tensor.extract_slice %[[OUT_INIT]][%arg3, %arg4] [32, 32] [1, 1]
//      CHECK:      %[[SLICE_OUT:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG]][%arg3, %arg4] [32, 32] [1, 1]
//      CHECK:      %[[ELEM_OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK-SAME:              ins(%[[MAT_OUT]], %[[SLICE_OPERAND2]] :
// CHECK-SAME:              outs(%[[SLICE_OUT]] :
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[ELEM_OUT]] into %[[ELEM_OUT_ARG]]
//      CHECK:          tensor.parallel_insert_slice %[[MAT_OUT]] into %[[MAT_OUT_ARG]]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   return %[[FINAL_RESULT]]#1 :
