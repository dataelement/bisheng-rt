�
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
:
Sub
x"T
y"T
z"T"
Ttype:
2	"serve*1.15.52unknown�
V
TENSOR_INPUT0Placeholder*
dtype0*
shape:*
_output_shapes
:
V
TENSOR_INPUT1Placeholder*
_output_shapes
:*
shape:*
dtype0
M
ADDAddTENSOR_INPUT0TENSOR_INPUT1*
T0*
_output_shapes
:
M
SUBSubTENSOR_INPUT0TENSOR_INPUT1*
_output_shapes
:*
T0
D
TENSOR_OUTPUT0IdentityADD*
T0*
_output_shapes
:
D
TENSOR_OUTPUT1IdentitySUB*
T0*
_output_shapes
: "�*�
serving_default�
#
INPUT0
TENSOR_INPUT0:0
#
INPUT1
TENSOR_INPUT1:0%
OUTPUT1
TENSOR_OUTPUT1:0%
OUTPUT0
TENSOR_OUTPUT0:0tensorflow/serving/predict