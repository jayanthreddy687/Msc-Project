��
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
cnn__encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_namecnn__encoder/dense/kernel
�
-cnn__encoder/dense/kernel/Read/ReadVariableOpReadVariableOpcnn__encoder/dense/kernel* 
_output_shapes
:
��*
dtype0
�
cnn__encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_namecnn__encoder/dense/bias
�
+cnn__encoder/dense/bias/Read/ReadVariableOpReadVariableOpcnn__encoder/dense/bias*
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
WQ
VARIABLE_VALUEcnn__encoder/dense/kernel$fc/kernel/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcnn__encoder/dense/bias"fc/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cnn__encoder/dense/kernelcnn__encoder/dense/bias*
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
%__inference_signature_wrapper_3070783
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-cnn__encoder/dense/kernel/Read/ReadVariableOp+cnn__encoder/dense/bias/Read/ReadVariableOpConst*
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
 __inference__traced_save_3070851
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn__encoder/dense/kernelcnn__encoder/dense/bias*
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
#__inference__traced_restore_3070867��
�
�
.__inference_cnn__encoder_layer_call_fn_3070703
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
GPU2*0J 8� *R
fMRK
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070696t
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
'__inference_dense_layer_call_fn_3070792

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
GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3070688t
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
 
_user_specified_nameinputs
� 
�
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070772
x;
'dense_tensordot_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       F
dense/Tensordot/ShapeShapex*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transposexdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������1��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������1�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������1�[
ReluReludense/BiasAdd:output:0*
T0*,
_output_shapes
:���������1�f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:���������1��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp:O K
,
_output_shapes
:���������1�

_user_specified_namex
�
�
B__inference_dense_layer_call_and_return_conditional_losses_3070822

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
B__inference_dense_layer_call_and_return_conditional_losses_3070688

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
�
�
 __inference__traced_save_3070851
file_prefix8
4savev2_cnn__encoder_dense_kernel_read_readvariableop6
2savev2_cnn__encoder_dense_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_cnn__encoder_dense_kernel_read_readvariableop2savev2_cnn__encoder_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�	
�
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070732
input_1!
dense_3070725:
��
dense_3070727:	�
identity��dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3070725dense_3070727*
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
GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3070688k
ReluRelu&dense/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:���������1�f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:���������1�f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:U Q
,
_output_shapes
:���������1�
!
_user_specified_name	input_1
�
�
%__inference_signature_wrapper_3070783
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
"__inference__wrapped_model_3070651t
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
#__inference__traced_restore_3070867
file_prefix>
*assignvariableop_cnn__encoder_dense_kernel:
��9
*assignvariableop_1_cnn__encoder_dense_bias:	�

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
AssignVariableOpAssignVariableOp*assignvariableop_cnn__encoder_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp*assignvariableop_1_cnn__encoder_dense_biasIdentity_1:output:0"/device:CPU:0*
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
�
�
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070696
x!
dense_3070689:
��
dense_3070691:	�
identity��dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallxdense_3070689dense_3070691*
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
GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3070688k
ReluRelu&dense/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:���������1�f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:���������1�f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
,
_output_shapes
:���������1�

_user_specified_namex
�&
�
"__inference__wrapped_model_3070651
input_1H
4cnn__encoder_dense_tensordot_readvariableop_resource:
��A
2cnn__encoder_dense_biasadd_readvariableop_resource:	�
identity��)cnn__encoder/dense/BiasAdd/ReadVariableOp�+cnn__encoder/dense/Tensordot/ReadVariableOp�
+cnn__encoder/dense/Tensordot/ReadVariableOpReadVariableOp4cnn__encoder_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0k
!cnn__encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!cnn__encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Y
"cnn__encoder/dense/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:l
*cnn__encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%cnn__encoder/dense/Tensordot/GatherV2GatherV2+cnn__encoder/dense/Tensordot/Shape:output:0*cnn__encoder/dense/Tensordot/free:output:03cnn__encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,cnn__encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'cnn__encoder/dense/Tensordot/GatherV2_1GatherV2+cnn__encoder/dense/Tensordot/Shape:output:0*cnn__encoder/dense/Tensordot/axes:output:05cnn__encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"cnn__encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
!cnn__encoder/dense/Tensordot/ProdProd.cnn__encoder/dense/Tensordot/GatherV2:output:0+cnn__encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$cnn__encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
#cnn__encoder/dense/Tensordot/Prod_1Prod0cnn__encoder/dense/Tensordot/GatherV2_1:output:0-cnn__encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(cnn__encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#cnn__encoder/dense/Tensordot/concatConcatV2*cnn__encoder/dense/Tensordot/free:output:0*cnn__encoder/dense/Tensordot/axes:output:01cnn__encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
"cnn__encoder/dense/Tensordot/stackPack*cnn__encoder/dense/Tensordot/Prod:output:0,cnn__encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
&cnn__encoder/dense/Tensordot/transpose	Transposeinput_1,cnn__encoder/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������1��
$cnn__encoder/dense/Tensordot/ReshapeReshape*cnn__encoder/dense/Tensordot/transpose:y:0+cnn__encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
#cnn__encoder/dense/Tensordot/MatMulMatMul-cnn__encoder/dense/Tensordot/Reshape:output:03cnn__encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
$cnn__encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�l
*cnn__encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%cnn__encoder/dense/Tensordot/concat_1ConcatV2.cnn__encoder/dense/Tensordot/GatherV2:output:0-cnn__encoder/dense/Tensordot/Const_2:output:03cnn__encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
cnn__encoder/dense/TensordotReshape-cnn__encoder/dense/Tensordot/MatMul:product:0.cnn__encoder/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������1��
)cnn__encoder/dense/BiasAdd/ReadVariableOpReadVariableOp2cnn__encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
cnn__encoder/dense/BiasAddBiasAdd%cnn__encoder/dense/Tensordot:output:01cnn__encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������1�u
cnn__encoder/ReluRelu#cnn__encoder/dense/BiasAdd:output:0*
T0*,
_output_shapes
:���������1�s
IdentityIdentitycnn__encoder/Relu:activations:0^NoOp*
T0*,
_output_shapes
:���������1��
NoOpNoOp*^cnn__encoder/dense/BiasAdd/ReadVariableOp,^cnn__encoder/dense/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������1�: : 2V
)cnn__encoder/dense/BiasAdd/ReadVariableOp)cnn__encoder/dense/BiasAdd/ReadVariableOp2Z
+cnn__encoder/dense/Tensordot/ReadVariableOp+cnn__encoder/dense/Tensordot/ReadVariableOp:U Q
,
_output_shapes
:���������1�
!
_user_specified_name	input_1
�
�
.__inference_cnn__encoder_layer_call_fn_3070741
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
GPU2*0J 8� *R
fMRK
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070696t
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
_user_specified_namex"�L
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
.__inference_cnn__encoder_layer_call_fn_3070703
.__inference_cnn__encoder_layer_call_fn_3070741�
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
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070772
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070732�
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
"__inference__wrapped_model_3070651input_1"�
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
-:+
��2cnn__encoder/dense/kernel
&:$�2cnn__encoder/dense/bias
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
'__inference_dense_layer_call_fn_3070792�
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
B__inference_dense_layer_call_and_return_conditional_losses_3070822�
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
%__inference_signature_wrapper_3070783input_1"�
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
"__inference__wrapped_model_3070651u
5�2
+�(
&�#
input_1���������1�
� "8�5
3
output_1'�$
output_1���������1��
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070732g
5�2
+�(
&�#
input_1���������1�
� "*�'
 �
0���������1�
� �
I__inference_cnn__encoder_layer_call_and_return_conditional_losses_3070772a
/�,
%�"
 �
x���������1�
� "*�'
 �
0���������1�
� �
.__inference_cnn__encoder_layer_call_fn_3070703Z
5�2
+�(
&�#
input_1���������1�
� "����������1��
.__inference_cnn__encoder_layer_call_fn_3070741T
/�,
%�"
 �
x���������1�
� "����������1��
B__inference_dense_layer_call_and_return_conditional_losses_3070822f
4�1
*�'
%�"
inputs���������1�
� "*�'
 �
0���������1�
� �
'__inference_dense_layer_call_fn_3070792Y
4�1
*�'
%�"
inputs���������1�
� "����������1��
%__inference_signature_wrapper_3070783�
@�=
� 
6�3
1
input_1&�#
input_1���������1�"8�5
3
output_1'�$
output_1���������1�