ҫ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��
�
cnn__encoder_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_namecnn__encoder_1/dense_6/kernel
�
1cnn__encoder_1/dense_6/kernel/Read/ReadVariableOpReadVariableOpcnn__encoder_1/dense_6/kernel* 
_output_shapes
:
��*
dtype0
�
cnn__encoder_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecnn__encoder_1/dense_6/bias
�
/cnn__encoder_1/dense_6/bias/Read/ReadVariableOpReadVariableOpcnn__encoder_1/dense_6/bias*
_output_shapes	
:�*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�

value�
B�
 B�

�
fc
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures*
�


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*


0
1*


0
1*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
[U
VARIABLE_VALUEcnn__encoder_1/dense_6/kernel$fc/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEcnn__encoder_1/dense_6/bias"fc/bias/.ATTRIBUTES/VARIABLE_VALUE*


0
1*


0
1*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
serving_default_input_1Placeholder*,
_output_shapes
:���������1�*
dtype0*!
shape:���������1�
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cnn__encoder_1/dense_6/kernelcnn__encoder_1/dense_6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������1�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_3598417
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1cnn__encoder_1/dense_6/kernel/Read/ReadVariableOp/cnn__encoder_1/dense_6/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_3598485
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn__encoder_1/dense_6/kernelcnn__encoder_1/dense_6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_3598501��
�	
�
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598330
x#
dense_6_3598323:
��
dense_6_3598325:	�
identity��dense_6/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallxdense_6_3598323dense_6_3598325*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������1�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_3598322m
ReluRelu(dense_6/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:���������1�f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:���������1�h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
,
_output_shapes
:���������1�

_user_specified_namex
�
�
%__inference_signature_wrapper_3598417
input_1
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������1�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_3598285t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������1�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:���������1�
!
_user_specified_name	input_1
�
�
#__inference__traced_restore_3598501
file_prefixB
.assignvariableop_cnn__encoder_1_dense_6_kernel:
��=
.assignvariableop_1_cnn__encoder_1_dense_6_bias:	�

identity_3��AssignVariableOp�AssignVariableOp_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*{
valuerBpB$fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp.assignvariableop_cnn__encoder_1_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_cnn__encoder_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: p
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*
_input_shapes
: : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�!
�
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598406
x=
)dense_6_tensordot_readvariableop_resource:
��6
'dense_6_biasadd_readvariableop_resource:	�
identity��dense_6/BiasAdd/ReadVariableOp� dense_6/Tensordot/ReadVariableOp�
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       H
dense_6/Tensordot/ShapeShapex*
T0*
_output_shapes
:a
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_6/Tensordot/transpose	Transposex!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������1��
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������1��
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������1�]
ReluReludense_6/BiasAdd:output:0*
T0*,
_output_shapes
:���������1�f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:���������1��
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp:O K
,
_output_shapes
:���������1�

_user_specified_namex
�
�
D__inference_dense_6_layer_call_and_return_conditional_losses_3598456

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������1��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������1�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������1�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������1�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������1�
 
_user_specified_nameinputs
�
�
D__inference_dense_6_layer_call_and_return_conditional_losses_3598322

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������1��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������1�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������1�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������1�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������1�
 
_user_specified_nameinputs
�	
�
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598366
input_1#
dense_6_3598359:
��
dense_6_3598361:	�
identity��dense_6/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6_3598359dense_6_3598361*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������1�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_3598322m
ReluRelu(dense_6/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:���������1�f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:���������1�h
NoOpNoOp ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:U Q
,
_output_shapes
:���������1�
!
_user_specified_name	input_1
�
�
 __inference__traced_save_3598485
file_prefix<
8savev2_cnn__encoder_1_dense_6_kernel_read_readvariableop:
6savev2_cnn__encoder_1_dense_6_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*{
valuerBpB$fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_cnn__encoder_1_dense_6_kernel_read_readvariableop6savev2_cnn__encoder_1_dense_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0**
_input_shapes
: :
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: 
�
�
0__inference_cnn__encoder_1_layer_call_fn_3598337
input_1
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������1�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598330t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������1�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:���������1�
!
_user_specified_name	input_1
�
�
0__inference_cnn__encoder_1_layer_call_fn_3598375
x
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������1�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598330t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������1�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:���������1�

_user_specified_namex
�(
�
"__inference__wrapped_model_3598285
input_1L
8cnn__encoder_1_dense_6_tensordot_readvariableop_resource:
��E
6cnn__encoder_1_dense_6_biasadd_readvariableop_resource:	�
identity��-cnn__encoder_1/dense_6/BiasAdd/ReadVariableOp�/cnn__encoder_1/dense_6/Tensordot/ReadVariableOp�
/cnn__encoder_1/dense_6/Tensordot/ReadVariableOpReadVariableOp8cnn__encoder_1_dense_6_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0o
%cnn__encoder_1/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%cnn__encoder_1/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ]
&cnn__encoder_1/dense_6/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:p
.cnn__encoder_1/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)cnn__encoder_1/dense_6/Tensordot/GatherV2GatherV2/cnn__encoder_1/dense_6/Tensordot/Shape:output:0.cnn__encoder_1/dense_6/Tensordot/free:output:07cnn__encoder_1/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0cnn__encoder_1/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+cnn__encoder_1/dense_6/Tensordot/GatherV2_1GatherV2/cnn__encoder_1/dense_6/Tensordot/Shape:output:0.cnn__encoder_1/dense_6/Tensordot/axes:output:09cnn__encoder_1/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&cnn__encoder_1/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%cnn__encoder_1/dense_6/Tensordot/ProdProd2cnn__encoder_1/dense_6/Tensordot/GatherV2:output:0/cnn__encoder_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(cnn__encoder_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'cnn__encoder_1/dense_6/Tensordot/Prod_1Prod4cnn__encoder_1/dense_6/Tensordot/GatherV2_1:output:01cnn__encoder_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,cnn__encoder_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'cnn__encoder_1/dense_6/Tensordot/concatConcatV2.cnn__encoder_1/dense_6/Tensordot/free:output:0.cnn__encoder_1/dense_6/Tensordot/axes:output:05cnn__encoder_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
&cnn__encoder_1/dense_6/Tensordot/stackPack.cnn__encoder_1/dense_6/Tensordot/Prod:output:00cnn__encoder_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
*cnn__encoder_1/dense_6/Tensordot/transpose	Transposeinput_10cnn__encoder_1/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������1��
(cnn__encoder_1/dense_6/Tensordot/ReshapeReshape.cnn__encoder_1/dense_6/Tensordot/transpose:y:0/cnn__encoder_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
'cnn__encoder_1/dense_6/Tensordot/MatMulMatMul1cnn__encoder_1/dense_6/Tensordot/Reshape:output:07cnn__encoder_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
(cnn__encoder_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�p
.cnn__encoder_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)cnn__encoder_1/dense_6/Tensordot/concat_1ConcatV22cnn__encoder_1/dense_6/Tensordot/GatherV2:output:01cnn__encoder_1/dense_6/Tensordot/Const_2:output:07cnn__encoder_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
 cnn__encoder_1/dense_6/TensordotReshape1cnn__encoder_1/dense_6/Tensordot/MatMul:product:02cnn__encoder_1/dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������1��
-cnn__encoder_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp6cnn__encoder_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
cnn__encoder_1/dense_6/BiasAddBiasAdd)cnn__encoder_1/dense_6/Tensordot:output:05cnn__encoder_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������1�{
cnn__encoder_1/ReluRelu'cnn__encoder_1/dense_6/BiasAdd:output:0*
T0*,
_output_shapes
:���������1�u
IdentityIdentity!cnn__encoder_1/Relu:activations:0^NoOp*
T0*,
_output_shapes
:���������1��
NoOpNoOp.^cnn__encoder_1/dense_6/BiasAdd/ReadVariableOp0^cnn__encoder_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 2^
-cnn__encoder_1/dense_6/BiasAdd/ReadVariableOp-cnn__encoder_1/dense_6/BiasAdd/ReadVariableOp2b
/cnn__encoder_1/dense_6/Tensordot/ReadVariableOp/cnn__encoder_1/dense_6/Tensordot/ReadVariableOp:U Q
,
_output_shapes
:���������1�
!
_user_specified_name	input_1
�
�
)__inference_dense_6_layer_call_fn_3598426

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������1�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_3598322t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������1�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������1�
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
@
input_15
serving_default_input_1:0���������1�A
output_15
StatefulPartitionedCall:0���������1�tensorflow/serving/predict:�#
�
fc
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures"
_tf_keras_model
�


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
0__inference_cnn__encoder_1_layer_call_fn_3598337
0__inference_cnn__encoder_1_layer_call_fn_3598375�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598406
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598366�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference__wrapped_model_3598285input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
serving_default"
signature_map
1:/
��2cnn__encoder_1/dense_6/kernel
*:(�2cnn__encoder_1/dense_6/bias
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_dense_6_layer_call_fn_3598426�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_6_layer_call_and_return_conditional_losses_3598456�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_signature_wrapper_3598417input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
"__inference__wrapped_model_3598285u
5�2
+�(
&�#
input_1���������1�
� "8�5
3
output_1'�$
output_1���������1��
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598366g
5�2
+�(
&�#
input_1���������1�
� "*�'
 �
0���������1�
� �
K__inference_cnn__encoder_1_layer_call_and_return_conditional_losses_3598406a
/�,
%�"
 �
x���������1�
� "*�'
 �
0���������1�
� �
0__inference_cnn__encoder_1_layer_call_fn_3598337Z
5�2
+�(
&�#
input_1���������1�
� "����������1��
0__inference_cnn__encoder_1_layer_call_fn_3598375T
/�,
%�"
 �
x���������1�
� "����������1��
D__inference_dense_6_layer_call_and_return_conditional_losses_3598456f
4�1
*�'
%�"
inputs���������1�
� "*�'
 �
0���������1�
� �
)__inference_dense_6_layer_call_fn_3598426Y
4�1
*�'
%�"
inputs���������1�
� "����������1��
%__inference_signature_wrapper_3598417�
@�=
� 
6�3
1
input_1&�#
input_1���������1�"8�5
3
output_1'�$
output_1���������1�