??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
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
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
Prot_CNN_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameProt_CNN_0/kernel
{
%Prot_CNN_0/kernel/Read/ReadVariableOpReadVariableOpProt_CNN_0/kernel*"
_output_shapes
:@*
dtype0
v
Prot_CNN_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameProt_CNN_0/bias
o
#Prot_CNN_0/bias/Read/ReadVariableOpReadVariableOpProt_CNN_0/bias*
_output_shapes
:@*
dtype0
?
SMILES_CNN_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameSMILES_CNN_0/kernel

'SMILES_CNN_0/kernel/Read/ReadVariableOpReadVariableOpSMILES_CNN_0/kernel*"
_output_shapes
:@*
dtype0
z
SMILES_CNN_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameSMILES_CNN_0/bias
s
%SMILES_CNN_0/bias/Read/ReadVariableOpReadVariableOpSMILES_CNN_0/bias*
_output_shapes
:@*
dtype0
?
Prot_CNN_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*"
shared_nameProt_CNN_1/kernel
{
%Prot_CNN_1/kernel/Read/ReadVariableOpReadVariableOpProt_CNN_1/kernel*"
_output_shapes
:@@*
dtype0
v
Prot_CNN_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameProt_CNN_1/bias
o
#Prot_CNN_1/bias/Read/ReadVariableOpReadVariableOpProt_CNN_1/bias*
_output_shapes
:@*
dtype0
?
SMILES_CNN_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameSMILES_CNN_1/kernel

'SMILES_CNN_1/kernel/Read/ReadVariableOpReadVariableOpSMILES_CNN_1/kernel*"
_output_shapes
:@@*
dtype0
z
SMILES_CNN_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameSMILES_CNN_1/bias
s
%SMILES_CNN_1/bias/Read/ReadVariableOpReadVariableOpSMILES_CNN_1/bias*
_output_shapes
:@*
dtype0
?
Prot_CNN_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*"
shared_nameProt_CNN_2/kernel
|
%Prot_CNN_2/kernel/Read/ReadVariableOpReadVariableOpProt_CNN_2/kernel*#
_output_shapes
:@?*
dtype0
w
Prot_CNN_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameProt_CNN_2/bias
p
#Prot_CNN_2/bias/Read/ReadVariableOpReadVariableOpProt_CNN_2/bias*
_output_shapes	
:?*
dtype0
?
SMILES_CNN_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameSMILES_CNN_2/kernel
?
'SMILES_CNN_2/kernel/Read/ReadVariableOpReadVariableOpSMILES_CNN_2/kernel*#
_output_shapes
:@?*
dtype0
{
SMILES_CNN_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameSMILES_CNN_2/bias
t
%SMILES_CNN_2/bias/Read/ReadVariableOpReadVariableOpSMILES_CNN_2/bias*
_output_shapes	
:?*
dtype0
z
Dense_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameDense_0/kernel
s
"Dense_0/kernel/Read/ReadVariableOpReadVariableOpDense_0/kernel* 
_output_shapes
:
??*
dtype0
q
Dense_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_0/bias
j
 Dense_0/bias/Read/ReadVariableOpReadVariableOpDense_0/bias*
_output_shapes	
:?*
dtype0
z
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameDense_1/kernel
s
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel* 
_output_shapes
:
??*
dtype0
q
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_1/bias
j
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes	
:?*
dtype0
z
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameDense_2/kernel
s
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel* 
_output_shapes
:
??*
dtype0
q
Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_2/bias
j
 Dense_2/bias/Read/ReadVariableOpReadVariableOpDense_2/bias*
_output_shapes	
:?*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Prot_CNN_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameProt_CNN_0/kernel/m

'Prot_CNN_0/kernel/m/Read/ReadVariableOpReadVariableOpProt_CNN_0/kernel/m*"
_output_shapes
:@*
dtype0
z
Prot_CNN_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameProt_CNN_0/bias/m
s
%Prot_CNN_0/bias/m/Read/ReadVariableOpReadVariableOpProt_CNN_0/bias/m*
_output_shapes
:@*
dtype0
?
SMILES_CNN_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameSMILES_CNN_0/kernel/m
?
)SMILES_CNN_0/kernel/m/Read/ReadVariableOpReadVariableOpSMILES_CNN_0/kernel/m*"
_output_shapes
:@*
dtype0
~
SMILES_CNN_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameSMILES_CNN_0/bias/m
w
'SMILES_CNN_0/bias/m/Read/ReadVariableOpReadVariableOpSMILES_CNN_0/bias/m*
_output_shapes
:@*
dtype0
?
Prot_CNN_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameProt_CNN_1/kernel/m

'Prot_CNN_1/kernel/m/Read/ReadVariableOpReadVariableOpProt_CNN_1/kernel/m*"
_output_shapes
:@@*
dtype0
z
Prot_CNN_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameProt_CNN_1/bias/m
s
%Prot_CNN_1/bias/m/Read/ReadVariableOpReadVariableOpProt_CNN_1/bias/m*
_output_shapes
:@*
dtype0
?
SMILES_CNN_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_nameSMILES_CNN_1/kernel/m
?
)SMILES_CNN_1/kernel/m/Read/ReadVariableOpReadVariableOpSMILES_CNN_1/kernel/m*"
_output_shapes
:@@*
dtype0
~
SMILES_CNN_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameSMILES_CNN_1/bias/m
w
'SMILES_CNN_1/bias/m/Read/ReadVariableOpReadVariableOpSMILES_CNN_1/bias/m*
_output_shapes
:@*
dtype0
?
Prot_CNN_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameProt_CNN_2/kernel/m
?
'Prot_CNN_2/kernel/m/Read/ReadVariableOpReadVariableOpProt_CNN_2/kernel/m*#
_output_shapes
:@?*
dtype0
{
Prot_CNN_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameProt_CNN_2/bias/m
t
%Prot_CNN_2/bias/m/Read/ReadVariableOpReadVariableOpProt_CNN_2/bias/m*
_output_shapes	
:?*
dtype0
?
SMILES_CNN_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*&
shared_nameSMILES_CNN_2/kernel/m
?
)SMILES_CNN_2/kernel/m/Read/ReadVariableOpReadVariableOpSMILES_CNN_2/kernel/m*#
_output_shapes
:@?*
dtype0

SMILES_CNN_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameSMILES_CNN_2/bias/m
x
'SMILES_CNN_2/bias/m/Read/ReadVariableOpReadVariableOpSMILES_CNN_2/bias/m*
_output_shapes	
:?*
dtype0
~
Dense_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameDense_0/kernel/m
w
$Dense_0/kernel/m/Read/ReadVariableOpReadVariableOpDense_0/kernel/m* 
_output_shapes
:
??*
dtype0
u
Dense_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_0/bias/m
n
"Dense_0/bias/m/Read/ReadVariableOpReadVariableOpDense_0/bias/m*
_output_shapes	
:?*
dtype0
~
Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameDense_1/kernel/m
w
$Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpDense_1/kernel/m* 
_output_shapes
:
??*
dtype0
u
Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_1/bias/m
n
"Dense_1/bias/m/Read/ReadVariableOpReadVariableOpDense_1/bias/m*
_output_shapes	
:?*
dtype0
~
Dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameDense_2/kernel/m
w
$Dense_2/kernel/m/Read/ReadVariableOpReadVariableOpDense_2/kernel/m* 
_output_shapes
:
??*
dtype0
u
Dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_2/bias/m
n
"Dense_2/bias/m/Read/ReadVariableOpReadVariableOpDense_2/bias/m*
_output_shapes	
:?*
dtype0
y
dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel/m
r
"dense/kernel/m/Read/ReadVariableOpReadVariableOpdense/kernel/m*
_output_shapes
:	?*
dtype0
p
dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias/m
i
 dense/bias/m/Read/ReadVariableOpReadVariableOpdense/bias/m*
_output_shapes
:*
dtype0
?
Prot_CNN_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameProt_CNN_0/kernel/v

'Prot_CNN_0/kernel/v/Read/ReadVariableOpReadVariableOpProt_CNN_0/kernel/v*"
_output_shapes
:@*
dtype0
z
Prot_CNN_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameProt_CNN_0/bias/v
s
%Prot_CNN_0/bias/v/Read/ReadVariableOpReadVariableOpProt_CNN_0/bias/v*
_output_shapes
:@*
dtype0
?
SMILES_CNN_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameSMILES_CNN_0/kernel/v
?
)SMILES_CNN_0/kernel/v/Read/ReadVariableOpReadVariableOpSMILES_CNN_0/kernel/v*"
_output_shapes
:@*
dtype0
~
SMILES_CNN_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameSMILES_CNN_0/bias/v
w
'SMILES_CNN_0/bias/v/Read/ReadVariableOpReadVariableOpSMILES_CNN_0/bias/v*
_output_shapes
:@*
dtype0
?
Prot_CNN_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameProt_CNN_1/kernel/v

'Prot_CNN_1/kernel/v/Read/ReadVariableOpReadVariableOpProt_CNN_1/kernel/v*"
_output_shapes
:@@*
dtype0
z
Prot_CNN_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameProt_CNN_1/bias/v
s
%Prot_CNN_1/bias/v/Read/ReadVariableOpReadVariableOpProt_CNN_1/bias/v*
_output_shapes
:@*
dtype0
?
SMILES_CNN_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_nameSMILES_CNN_1/kernel/v
?
)SMILES_CNN_1/kernel/v/Read/ReadVariableOpReadVariableOpSMILES_CNN_1/kernel/v*"
_output_shapes
:@@*
dtype0
~
SMILES_CNN_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameSMILES_CNN_1/bias/v
w
'SMILES_CNN_1/bias/v/Read/ReadVariableOpReadVariableOpSMILES_CNN_1/bias/v*
_output_shapes
:@*
dtype0
?
Prot_CNN_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameProt_CNN_2/kernel/v
?
'Prot_CNN_2/kernel/v/Read/ReadVariableOpReadVariableOpProt_CNN_2/kernel/v*#
_output_shapes
:@?*
dtype0
{
Prot_CNN_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameProt_CNN_2/bias/v
t
%Prot_CNN_2/bias/v/Read/ReadVariableOpReadVariableOpProt_CNN_2/bias/v*
_output_shapes	
:?*
dtype0
?
SMILES_CNN_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*&
shared_nameSMILES_CNN_2/kernel/v
?
)SMILES_CNN_2/kernel/v/Read/ReadVariableOpReadVariableOpSMILES_CNN_2/kernel/v*#
_output_shapes
:@?*
dtype0

SMILES_CNN_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameSMILES_CNN_2/bias/v
x
'SMILES_CNN_2/bias/v/Read/ReadVariableOpReadVariableOpSMILES_CNN_2/bias/v*
_output_shapes	
:?*
dtype0
~
Dense_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameDense_0/kernel/v
w
$Dense_0/kernel/v/Read/ReadVariableOpReadVariableOpDense_0/kernel/v* 
_output_shapes
:
??*
dtype0
u
Dense_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_0/bias/v
n
"Dense_0/bias/v/Read/ReadVariableOpReadVariableOpDense_0/bias/v*
_output_shapes	
:?*
dtype0
~
Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameDense_1/kernel/v
w
$Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpDense_1/kernel/v* 
_output_shapes
:
??*
dtype0
u
Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_1/bias/v
n
"Dense_1/bias/v/Read/ReadVariableOpReadVariableOpDense_1/bias/v*
_output_shapes	
:?*
dtype0
~
Dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameDense_2/kernel/v
w
$Dense_2/kernel/v/Read/ReadVariableOpReadVariableOpDense_2/kernel/v* 
_output_shapes
:
??*
dtype0
u
Dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense_2/bias/v
n
"Dense_2/bias/v/Read/ReadVariableOpReadVariableOpDense_2/bias/v*
_output_shapes	
:?*
dtype0
y
dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel/v
r
"dense/kernel/v/Read/ReadVariableOpReadVariableOpdense/kernel/v*
_output_shapes
:	?*
dtype0
p
dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense/bias/v
i
 dense/bias/v/Read/ReadVariableOpReadVariableOpdense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?q
value?qB?q B?q
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
R
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
h

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
R
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
h

\kernel
]bias
^regularization_losses
_	variables
`trainable_variables
a	keras_api
R
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
h

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
h

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Rm?Sm?\m?]m?fm?gm?lm?mm?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Rv?Sv?\v?]v?fv?gv?lv?mv?
 
?
"0
#1
(2
)3
.4
/5
46
57
:8
;9
@10
A11
R12
S13
\14
]15
f16
g17
l18
m19
?
"0
#1
(2
)3
.4
/5
46
57
:8
;9
@10
A11
R12
S13
\14
]15
f16
g17
l18
m19
?
rlayer_metrics

slayers
tlayer_regularization_losses
umetrics
regularization_losses
	variables
trainable_variables
vnon_trainable_variables
 
 
 
 
?
wnon_trainable_variables

xlayers
ylayer_regularization_losses
zmetrics
regularization_losses
	variables
trainable_variables
{layer_metrics
 
 
 
?
|non_trainable_variables

}layers
~layer_regularization_losses
metrics
regularization_losses
	variables
 trainable_variables
?layer_metrics
][
VARIABLE_VALUEProt_CNN_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEProt_CNN_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
$regularization_losses
%	variables
&trainable_variables
?layer_metrics
_]
VARIABLE_VALUESMILES_CNN_0/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUESMILES_CNN_0/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
*regularization_losses
+	variables
,trainable_variables
?layer_metrics
][
VARIABLE_VALUEProt_CNN_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEProt_CNN_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
0regularization_losses
1	variables
2trainable_variables
?layer_metrics
_]
VARIABLE_VALUESMILES_CNN_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUESMILES_CNN_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
6regularization_losses
7	variables
8trainable_variables
?layer_metrics
][
VARIABLE_VALUEProt_CNN_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEProt_CNN_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
<regularization_losses
=	variables
>trainable_variables
?layer_metrics
_]
VARIABLE_VALUESMILES_CNN_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUESMILES_CNN_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Bregularization_losses
C	variables
Dtrainable_variables
?layer_metrics
 
 
 
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Fregularization_losses
G	variables
Htrainable_variables
?layer_metrics
 
 
 
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Jregularization_losses
K	variables
Ltrainable_variables
?layer_metrics
 
 
 
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Nregularization_losses
O	variables
Ptrainable_variables
?layer_metrics
ZX
VARIABLE_VALUEDense_0/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_0/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Tregularization_losses
U	variables
Vtrainable_variables
?layer_metrics
 
 
 
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Xregularization_losses
Y	variables
Ztrainable_variables
?layer_metrics
ZX
VARIABLE_VALUEDense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

\0
]1

\0
]1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
^regularization_losses
_	variables
`trainable_variables
?layer_metrics
 
 
 
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
bregularization_losses
c	variables
dtrainable_variables
?layer_metrics
ZX
VARIABLE_VALUEDense_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1

f0
g1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
hregularization_losses
i	variables
jtrainable_variables
?layer_metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

l0
m1
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
nregularization_losses
o	variables
ptrainable_variables
?layer_metrics
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
 

?0
?1
?2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
{y
VARIABLE_VALUEProt_CNN_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEProt_CNN_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUESMILES_CNN_0/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESMILES_CNN_0/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEProt_CNN_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEProt_CNN_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUESMILES_CNN_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESMILES_CNN_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEProt_CNN_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEProt_CNN_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUESMILES_CNN_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESMILES_CNN_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEDense_0/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEDense_0/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEDense_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEDense_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEDense_2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEDense_2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEProt_CNN_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEProt_CNN_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUESMILES_CNN_0/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESMILES_CNN_0/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEProt_CNN_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEProt_CNN_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUESMILES_CNN_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESMILES_CNN_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEProt_CNN_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEProt_CNN_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUESMILES_CNN_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESMILES_CNN_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEDense_0/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEDense_0/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEDense_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEDense_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEDense_2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEDense_2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_Protein_InputPlaceholder*(
_output_shapes
:??????????
*
dtype0	*
shape:??????????


serving_default_SMILES_InputPlaceholder*'
_output_shapes
:?????????H*
dtype0	*
shape:?????????H
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_Protein_Inputserving_default_SMILES_InputSMILES_CNN_0/kernelSMILES_CNN_0/biasProt_CNN_0/kernelProt_CNN_0/biasSMILES_CNN_1/kernelSMILES_CNN_1/biasProt_CNN_1/kernelProt_CNN_1/biasSMILES_CNN_2/kernelSMILES_CNN_2/biasProt_CNN_2/kernelProt_CNN_2/biasDense_0/kernelDense_0/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasdense/kernel
dense/bias*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_3201
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%Prot_CNN_0/kernel/Read/ReadVariableOp#Prot_CNN_0/bias/Read/ReadVariableOp'SMILES_CNN_0/kernel/Read/ReadVariableOp%SMILES_CNN_0/bias/Read/ReadVariableOp%Prot_CNN_1/kernel/Read/ReadVariableOp#Prot_CNN_1/bias/Read/ReadVariableOp'SMILES_CNN_1/kernel/Read/ReadVariableOp%SMILES_CNN_1/bias/Read/ReadVariableOp%Prot_CNN_2/kernel/Read/ReadVariableOp#Prot_CNN_2/bias/Read/ReadVariableOp'SMILES_CNN_2/kernel/Read/ReadVariableOp%SMILES_CNN_2/bias/Read/ReadVariableOp"Dense_0/kernel/Read/ReadVariableOp Dense_0/bias/Read/ReadVariableOp"Dense_1/kernel/Read/ReadVariableOp Dense_1/bias/Read/ReadVariableOp"Dense_2/kernel/Read/ReadVariableOp Dense_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp'Prot_CNN_0/kernel/m/Read/ReadVariableOp%Prot_CNN_0/bias/m/Read/ReadVariableOp)SMILES_CNN_0/kernel/m/Read/ReadVariableOp'SMILES_CNN_0/bias/m/Read/ReadVariableOp'Prot_CNN_1/kernel/m/Read/ReadVariableOp%Prot_CNN_1/bias/m/Read/ReadVariableOp)SMILES_CNN_1/kernel/m/Read/ReadVariableOp'SMILES_CNN_1/bias/m/Read/ReadVariableOp'Prot_CNN_2/kernel/m/Read/ReadVariableOp%Prot_CNN_2/bias/m/Read/ReadVariableOp)SMILES_CNN_2/kernel/m/Read/ReadVariableOp'SMILES_CNN_2/bias/m/Read/ReadVariableOp$Dense_0/kernel/m/Read/ReadVariableOp"Dense_0/bias/m/Read/ReadVariableOp$Dense_1/kernel/m/Read/ReadVariableOp"Dense_1/bias/m/Read/ReadVariableOp$Dense_2/kernel/m/Read/ReadVariableOp"Dense_2/bias/m/Read/ReadVariableOp"dense/kernel/m/Read/ReadVariableOp dense/bias/m/Read/ReadVariableOp'Prot_CNN_0/kernel/v/Read/ReadVariableOp%Prot_CNN_0/bias/v/Read/ReadVariableOp)SMILES_CNN_0/kernel/v/Read/ReadVariableOp'SMILES_CNN_0/bias/v/Read/ReadVariableOp'Prot_CNN_1/kernel/v/Read/ReadVariableOp%Prot_CNN_1/bias/v/Read/ReadVariableOp)SMILES_CNN_1/kernel/v/Read/ReadVariableOp'SMILES_CNN_1/bias/v/Read/ReadVariableOp'Prot_CNN_2/kernel/v/Read/ReadVariableOp%Prot_CNN_2/bias/v/Read/ReadVariableOp)SMILES_CNN_2/kernel/v/Read/ReadVariableOp'SMILES_CNN_2/bias/v/Read/ReadVariableOp$Dense_0/kernel/v/Read/ReadVariableOp"Dense_0/bias/v/Read/ReadVariableOp$Dense_1/kernel/v/Read/ReadVariableOp"Dense_1/bias/v/Read/ReadVariableOp$Dense_2/kernel/v/Read/ReadVariableOp"Dense_2/bias/v/Read/ReadVariableOp"dense/kernel/v/Read/ReadVariableOp dense/bias/v/Read/ReadVariableOpConst*O
TinH
F2D*
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
GPU2*0J 8? *&
f!R
__inference__traced_save_4149
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameProt_CNN_0/kernelProt_CNN_0/biasSMILES_CNN_0/kernelSMILES_CNN_0/biasProt_CNN_1/kernelProt_CNN_1/biasSMILES_CNN_1/kernelSMILES_CNN_1/biasProt_CNN_2/kernelProt_CNN_2/biasSMILES_CNN_2/kernelSMILES_CNN_2/biasDense_0/kernelDense_0/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasdense/kernel
dense/biastotalcounttotal_1count_1total_2count_2Prot_CNN_0/kernel/mProt_CNN_0/bias/mSMILES_CNN_0/kernel/mSMILES_CNN_0/bias/mProt_CNN_1/kernel/mProt_CNN_1/bias/mSMILES_CNN_1/kernel/mSMILES_CNN_1/bias/mProt_CNN_2/kernel/mProt_CNN_2/bias/mSMILES_CNN_2/kernel/mSMILES_CNN_2/bias/mDense_0/kernel/mDense_0/bias/mDense_1/kernel/mDense_1/bias/mDense_2/kernel/mDense_2/bias/mdense/kernel/mdense/bias/mProt_CNN_0/kernel/vProt_CNN_0/bias/vSMILES_CNN_0/kernel/vSMILES_CNN_0/bias/vProt_CNN_1/kernel/vProt_CNN_1/bias/vSMILES_CNN_1/kernel/vSMILES_CNN_1/bias/vProt_CNN_2/kernel/vProt_CNN_2/bias/vSMILES_CNN_2/kernel/vSMILES_CNN_2/bias/vDense_0/kernel/vDense_0/bias/vDense_1/kernel/vDense_1/bias/vDense_2/kernel/vDense_2/bias/vdense/kernel/vdense/bias/v*N
TinG
E2C*
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
GPU2*0J 8? *)
f$R"
 __inference__traced_restore_4357̬
?
?
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_3647

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????
@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
`
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_2446

inputs	
identity]
CastCastinputs*

DstT0*

SrcT0	*'
_output_shapes
:?????????H2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????H2	
one_hot?
GatherV2/indicesConst*
_output_shapes
:*
dtype0*}
valuetBr"h                        	   
                                                   2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2one_hot:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????H2

GatherV2i
IdentityIdentityGatherV2:output:0*
T0*+
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????H:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
b
C__inference_Dropout_1_layer_call_and_return_conditional_losses_2802

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_Dropout_0_layer_call_and_return_conditional_losses_3831

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_2640

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????H?*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????H?*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????H?2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????H?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????H@
 
_user_specified_nameinputs
?	
?
A__inference_Dense_1_layer_call_and_return_conditional_losses_3852

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_3343
inputs_0	
inputs_1	<
8smiles_cnn_0_conv1d_expanddims_1_readvariableop_resource0
,smiles_cnn_0_biasadd_readvariableop_resource:
6prot_cnn_0_conv1d_expanddims_1_readvariableop_resource.
*prot_cnn_0_biasadd_readvariableop_resource<
8smiles_cnn_1_conv1d_expanddims_1_readvariableop_resource0
,smiles_cnn_1_biasadd_readvariableop_resource:
6prot_cnn_1_conv1d_expanddims_1_readvariableop_resource.
*prot_cnn_1_biasadd_readvariableop_resource<
8smiles_cnn_2_conv1d_expanddims_1_readvariableop_resource0
,smiles_cnn_2_biasadd_readvariableop_resource:
6prot_cnn_2_conv1d_expanddims_1_readvariableop_resource.
*prot_cnn_2_biasadd_readvariableop_resource*
&dense_0_matmul_readvariableop_resource+
'dense_0_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??Dense_0/BiasAdd/ReadVariableOp?Dense_0/MatMul/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?!Prot_CNN_0/BiasAdd/ReadVariableOp?-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?!Prot_CNN_1/BiasAdd/ReadVariableOp?-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?!Prot_CNN_2/BiasAdd/ReadVariableOp?-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?#SMILES_CNN_0/BiasAdd/ReadVariableOp?/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?#SMILES_CNN_1/BiasAdd/ReadVariableOp?/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?#SMILES_CNN_2/BiasAdd/ReadVariableOp?/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpu
SMILES_Emb/CastCastinputs_1*

DstT0*

SrcT0	*'
_output_shapes
:?????????H2
SMILES_Emb/Cast
SMILES_Emb/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
SMILES_Emb/one_hot/on_value?
SMILES_Emb/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SMILES_Emb/one_hot/off_valuev
SMILES_Emb/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
SMILES_Emb/one_hot/depth?
SMILES_Emb/one_hotOneHotSMILES_Emb/Cast:y:0!SMILES_Emb/one_hot/depth:output:0$SMILES_Emb/one_hot/on_value:output:0%SMILES_Emb/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????H2
SMILES_Emb/one_hot?
SMILES_Emb/GatherV2/indicesConst*
_output_shapes
:*
dtype0*}
valuetBr"h                        	   
                                                   2
SMILES_Emb/GatherV2/indicesv
SMILES_Emb/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
SMILES_Emb/GatherV2/axis?
SMILES_Emb/GatherV2GatherV2SMILES_Emb/one_hot:output:0$SMILES_Emb/GatherV2/indices:output:0!SMILES_Emb/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????H2
SMILES_Emb/GatherV2r
Prot_Emb/CastCastinputs_0*

DstT0*

SrcT0	*(
_output_shapes
:??????????
2
Prot_Emb/Cast{
Prot_Emb/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
Prot_Emb/one_hot/on_value}
Prot_Emb/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Prot_Emb/one_hot/off_valuer
Prot_Emb/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
Prot_Emb/one_hot/depth?
Prot_Emb/one_hotOneHotProt_Emb/Cast:y:0Prot_Emb/one_hot/depth:output:0"Prot_Emb/one_hot/on_value:output:0#Prot_Emb/one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:??????????
2
Prot_Emb/one_hot?
Prot_Emb/GatherV2/indicesConst*
_output_shapes
:*
dtype0*e
value\BZ"P                        	   
                                 2
Prot_Emb/GatherV2/indicesr
Prot_Emb/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Prot_Emb/GatherV2/axis?
Prot_Emb/GatherV2GatherV2Prot_Emb/one_hot:output:0"Prot_Emb/GatherV2/indices:output:0Prot_Emb/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:??????????
2
Prot_Emb/GatherV2?
"SMILES_CNN_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"SMILES_CNN_0/conv1d/ExpandDims/dim?
SMILES_CNN_0/conv1d/ExpandDims
ExpandDimsSMILES_Emb/GatherV2:output:0+SMILES_CNN_0/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H2 
SMILES_CNN_0/conv1d/ExpandDims?
/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8smiles_cnn_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype021
/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?
$SMILES_CNN_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$SMILES_CNN_0/conv1d/ExpandDims_1/dim?
 SMILES_CNN_0/conv1d/ExpandDims_1
ExpandDims7SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp:value:0-SMILES_CNN_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2"
 SMILES_CNN_0/conv1d/ExpandDims_1?
SMILES_CNN_0/conv1dConv2D'SMILES_CNN_0/conv1d/ExpandDims:output:0)SMILES_CNN_0/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2
SMILES_CNN_0/conv1d?
SMILES_CNN_0/conv1d/SqueezeSqueezeSMILES_CNN_0/conv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2
SMILES_CNN_0/conv1d/Squeeze?
#SMILES_CNN_0/BiasAdd/ReadVariableOpReadVariableOp,smiles_cnn_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#SMILES_CNN_0/BiasAdd/ReadVariableOp?
SMILES_CNN_0/BiasAddBiasAdd$SMILES_CNN_0/conv1d/Squeeze:output:0+SMILES_CNN_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2
SMILES_CNN_0/BiasAdd?
SMILES_CNN_0/ReluReluSMILES_CNN_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2
SMILES_CNN_0/Relu?
 Prot_CNN_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 Prot_CNN_0/conv1d/ExpandDims/dim?
Prot_CNN_0/conv1d/ExpandDims
ExpandDimsProt_Emb/GatherV2:output:0)Prot_CNN_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
Prot_CNN_0/conv1d/ExpandDims?
-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6prot_cnn_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02/
-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?
"Prot_CNN_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Prot_CNN_0/conv1d/ExpandDims_1/dim?
Prot_CNN_0/conv1d/ExpandDims_1
ExpandDims5Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp:value:0+Prot_CNN_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2 
Prot_CNN_0/conv1d/ExpandDims_1?
Prot_CNN_0/conv1dConv2D%Prot_CNN_0/conv1d/ExpandDims:output:0'Prot_CNN_0/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2
Prot_CNN_0/conv1d?
Prot_CNN_0/conv1d/SqueezeSqueezeProt_CNN_0/conv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2
Prot_CNN_0/conv1d/Squeeze?
!Prot_CNN_0/BiasAdd/ReadVariableOpReadVariableOp*prot_cnn_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Prot_CNN_0/BiasAdd/ReadVariableOp?
Prot_CNN_0/BiasAddBiasAdd"Prot_CNN_0/conv1d/Squeeze:output:0)Prot_CNN_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2
Prot_CNN_0/BiasAdd~
Prot_CNN_0/ReluReluProt_CNN_0/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2
Prot_CNN_0/Relu?
"SMILES_CNN_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"SMILES_CNN_1/conv1d/ExpandDims/dim?
SMILES_CNN_1/conv1d/ExpandDims
ExpandDimsSMILES_CNN_0/Relu:activations:0+SMILES_CNN_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2 
SMILES_CNN_1/conv1d/ExpandDims?
/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8smiles_cnn_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype021
/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?
$SMILES_CNN_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$SMILES_CNN_1/conv1d/ExpandDims_1/dim?
 SMILES_CNN_1/conv1d/ExpandDims_1
ExpandDims7SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp:value:0-SMILES_CNN_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2"
 SMILES_CNN_1/conv1d/ExpandDims_1?
SMILES_CNN_1/conv1dConv2D'SMILES_CNN_1/conv1d/ExpandDims:output:0)SMILES_CNN_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2
SMILES_CNN_1/conv1d?
SMILES_CNN_1/conv1d/SqueezeSqueezeSMILES_CNN_1/conv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2
SMILES_CNN_1/conv1d/Squeeze?
#SMILES_CNN_1/BiasAdd/ReadVariableOpReadVariableOp,smiles_cnn_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#SMILES_CNN_1/BiasAdd/ReadVariableOp?
SMILES_CNN_1/BiasAddBiasAdd$SMILES_CNN_1/conv1d/Squeeze:output:0+SMILES_CNN_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2
SMILES_CNN_1/BiasAdd?
SMILES_CNN_1/ReluReluSMILES_CNN_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2
SMILES_CNN_1/Relu?
 Prot_CNN_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 Prot_CNN_1/conv1d/ExpandDims/dim?
Prot_CNN_1/conv1d/ExpandDims
ExpandDimsProt_CNN_0/Relu:activations:0)Prot_CNN_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2
Prot_CNN_1/conv1d/ExpandDims?
-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6prot_cnn_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?
"Prot_CNN_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Prot_CNN_1/conv1d/ExpandDims_1/dim?
Prot_CNN_1/conv1d/ExpandDims_1
ExpandDims5Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp:value:0+Prot_CNN_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2 
Prot_CNN_1/conv1d/ExpandDims_1?
Prot_CNN_1/conv1dConv2D%Prot_CNN_1/conv1d/ExpandDims:output:0'Prot_CNN_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2
Prot_CNN_1/conv1d?
Prot_CNN_1/conv1d/SqueezeSqueezeProt_CNN_1/conv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2
Prot_CNN_1/conv1d/Squeeze?
!Prot_CNN_1/BiasAdd/ReadVariableOpReadVariableOp*prot_cnn_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Prot_CNN_1/BiasAdd/ReadVariableOp?
Prot_CNN_1/BiasAddBiasAdd"Prot_CNN_1/conv1d/Squeeze:output:0)Prot_CNN_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2
Prot_CNN_1/BiasAdd~
Prot_CNN_1/ReluReluProt_CNN_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2
Prot_CNN_1/Relu?
"SMILES_CNN_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"SMILES_CNN_2/conv1d/ExpandDims/dim?
SMILES_CNN_2/conv1d/ExpandDims
ExpandDimsSMILES_CNN_1/Relu:activations:0+SMILES_CNN_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2 
SMILES_CNN_2/conv1d/ExpandDims?
/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8smiles_cnn_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype021
/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?
$SMILES_CNN_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$SMILES_CNN_2/conv1d/ExpandDims_1/dim?
 SMILES_CNN_2/conv1d/ExpandDims_1
ExpandDims7SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp:value:0-SMILES_CNN_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2"
 SMILES_CNN_2/conv1d/ExpandDims_1?
SMILES_CNN_2/conv1dConv2D'SMILES_CNN_2/conv1d/ExpandDims:output:0)SMILES_CNN_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????H?*
paddingSAME*
strides
2
SMILES_CNN_2/conv1d?
SMILES_CNN_2/conv1d/SqueezeSqueezeSMILES_CNN_2/conv1d:output:0*
T0*,
_output_shapes
:?????????H?*
squeeze_dims

?????????2
SMILES_CNN_2/conv1d/Squeeze?
#SMILES_CNN_2/BiasAdd/ReadVariableOpReadVariableOp,smiles_cnn_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#SMILES_CNN_2/BiasAdd/ReadVariableOp?
SMILES_CNN_2/BiasAddBiasAdd$SMILES_CNN_2/conv1d/Squeeze:output:0+SMILES_CNN_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????H?2
SMILES_CNN_2/BiasAdd?
SMILES_CNN_2/ReluReluSMILES_CNN_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????H?2
SMILES_CNN_2/Relu?
 Prot_CNN_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 Prot_CNN_2/conv1d/ExpandDims/dim?
Prot_CNN_2/conv1d/ExpandDims
ExpandDimsProt_CNN_1/Relu:activations:0)Prot_CNN_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2
Prot_CNN_2/conv1d/ExpandDims?
-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6prot_cnn_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02/
-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?
"Prot_CNN_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Prot_CNN_2/conv1d/ExpandDims_1/dim?
Prot_CNN_2/conv1d/ExpandDims_1
ExpandDims5Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp:value:0+Prot_CNN_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2 
Prot_CNN_2/conv1d/ExpandDims_1?
Prot_CNN_2/conv1dConv2D%Prot_CNN_2/conv1d/ExpandDims:output:0'Prot_CNN_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:??????????
?*
paddingSAME*
strides
2
Prot_CNN_2/conv1d?
Prot_CNN_2/conv1d/SqueezeSqueezeProt_CNN_2/conv1d:output:0*
T0*-
_output_shapes
:??????????
?*
squeeze_dims

?????????2
Prot_CNN_2/conv1d/Squeeze?
!Prot_CNN_2/BiasAdd/ReadVariableOpReadVariableOp*prot_cnn_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!Prot_CNN_2/BiasAdd/ReadVariableOp?
Prot_CNN_2/BiasAddBiasAdd"Prot_CNN_2/conv1d/Squeeze:output:0)Prot_CNN_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????
?2
Prot_CNN_2/BiasAdd
Prot_CNN_2/ReluReluProt_CNN_2/BiasAdd:output:0*
T0*-
_output_shapes
:??????????
?2
Prot_CNN_2/Relu?
%Prot_Global_Max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%Prot_Global_Max/Max/reduction_indices?
Prot_Global_Max/MaxMaxProt_CNN_2/Relu:activations:0.Prot_Global_Max/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Prot_Global_Max/Max?
'SMILES_Global_Max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2)
'SMILES_Global_Max/Max/reduction_indices?
SMILES_Global_Max/MaxMaxSMILES_CNN_2/Relu:activations:00SMILES_Global_Max/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
SMILES_Global_Max/Maxt
Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate/concat/axis?
Concatenate/concatConcatV2Prot_Global_Max/Max:output:0SMILES_Global_Max/Max:output:0 Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
Concatenate/concat?
Dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_0/MatMul/ReadVariableOp?
Dense_0/MatMulMatMulConcatenate/concat:output:0%Dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_0/MatMul?
Dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_0/BiasAdd/ReadVariableOp?
Dense_0/BiasAddBiasAddDense_0/MatMul:product:0&Dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_0/BiasAddq
Dense_0/ReluReluDense_0/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_0/Reluw
Dropout_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Dropout_0/dropout/Const?
Dropout_0/dropout/MulMulDense_0/Relu:activations:0 Dropout_0/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
Dropout_0/dropout/Mul|
Dropout_0/dropout/ShapeShapeDense_0/Relu:activations:0*
T0*
_output_shapes
:2
Dropout_0/dropout/Shape?
.Dropout_0/dropout/random_uniform/RandomUniformRandomUniform Dropout_0/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.Dropout_0/dropout/random_uniform/RandomUniform?
 Dropout_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 Dropout_0/dropout/GreaterEqual/y?
Dropout_0/dropout/GreaterEqualGreaterEqual7Dropout_0/dropout/random_uniform/RandomUniform:output:0)Dropout_0/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
Dropout_0/dropout/GreaterEqual?
Dropout_0/dropout/CastCast"Dropout_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
Dropout_0/dropout/Cast?
Dropout_0/dropout/Mul_1MulDropout_0/dropout/Mul:z:0Dropout_0/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
Dropout_0/dropout/Mul_1?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMulDropout_0/dropout/Mul_1:z:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/BiasAddq
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_1/Reluw
Dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
Dropout_1/dropout/Const?
Dropout_1/dropout/MulMulDense_1/Relu:activations:0 Dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
Dropout_1/dropout/Mul|
Dropout_1/dropout/ShapeShapeDense_1/Relu:activations:0*
T0*
_output_shapes
:2
Dropout_1/dropout/Shape?
.Dropout_1/dropout/random_uniform/RandomUniformRandomUniform Dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.Dropout_1/dropout/random_uniform/RandomUniform?
 Dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 Dropout_1/dropout/GreaterEqual/y?
Dropout_1/dropout/GreaterEqualGreaterEqual7Dropout_1/dropout/random_uniform/RandomUniform:output:0)Dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
Dropout_1/dropout/GreaterEqual?
Dropout_1/dropout/CastCast"Dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
Dropout_1/dropout/Cast?
Dropout_1/dropout/Mul_1MulDropout_1/dropout/Mul:z:0Dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
Dropout_1/dropout/Mul_1?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMulDropout_1/dropout/Mul_1:z:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_2/BiasAddq
Dense_2/ReluReluDense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_2/Relu?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulDense_2/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^Dense_0/BiasAdd/ReadVariableOp^Dense_0/MatMul/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp"^Prot_CNN_0/BiasAdd/ReadVariableOp.^Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp"^Prot_CNN_1/BiasAdd/ReadVariableOp.^Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp"^Prot_CNN_2/BiasAdd/ReadVariableOp.^Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp$^SMILES_CNN_0/BiasAdd/ReadVariableOp0^SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp$^SMILES_CNN_1/BiasAdd/ReadVariableOp0^SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp$^SMILES_CNN_2/BiasAdd/ReadVariableOp0^SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::2@
Dense_0/BiasAdd/ReadVariableOpDense_0/BiasAdd/ReadVariableOp2>
Dense_0/MatMul/ReadVariableOpDense_0/MatMul/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp2F
!Prot_CNN_0/BiasAdd/ReadVariableOp!Prot_CNN_0/BiasAdd/ReadVariableOp2^
-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp2F
!Prot_CNN_1/BiasAdd/ReadVariableOp!Prot_CNN_1/BiasAdd/ReadVariableOp2^
-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp2F
!Prot_CNN_2/BiasAdd/ReadVariableOp!Prot_CNN_2/BiasAdd/ReadVariableOp2^
-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp2J
#SMILES_CNN_0/BiasAdd/ReadVariableOp#SMILES_CNN_0/BiasAdd/ReadVariableOp2b
/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp2J
#SMILES_CNN_1/BiasAdd/ReadVariableOp#SMILES_CNN_1/BiasAdd/ReadVariableOp2b
/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp2J
#SMILES_CNN_2/BiasAdd/ReadVariableOp#SMILES_CNN_2/BiasAdd/ReadVariableOp2b
/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????H
"
_user_specified_name
inputs/1
?
g
K__inference_SMILES_Global_Max_layer_call_and_return_conditional_losses_2411

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_2857

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_2483

inputs	
identity^
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????
2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:??????????
2	
one_hot?
GatherV2/indicesConst*
_output_shapes
:*
dtype0*e
value\BZ"P                        	   
                                 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2one_hot:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:??????????
2

GatherV2j
IdentityIdentityGatherV2:output:0*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
L
0__inference_SMILES_Global_Max_layer_call_fn_2417

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_SMILES_Global_Max_layer_call_and_return_conditional_losses_24112
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_3772

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????H?*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:?????????H?*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????H?2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????H?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????H@
 
_user_specified_nameinputs
?	
?
A__inference_Dense_2_layer_call_and_return_conditional_losses_3899

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_Dropout_0_layer_call_and_return_conditional_losses_2750

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_SMILES_Emb_layer_call_fn_3626

inputs	
identity?
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_24342
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????H:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
~
)__inference_Prot_CNN_0_layer_call_fn_3656

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_25442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????
@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
b
C__inference_Dropout_1_layer_call_and_return_conditional_losses_3873

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
q
E__inference_Concatenate_layer_call_and_return_conditional_losses_3788
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
"__inference_signature_wrapper_3201
protein_input	
smiles_input	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprotein_inputsmiles_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_23912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????

'
_user_specified_nameProtein_Input:UQ
'
_output_shapes
:?????????H
&
_user_specified_nameSMILES_Input
?
~
)__inference_Prot_CNN_1_layer_call_fn_3706

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_26082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????
@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
@
 
_user_specified_nameinputs
?
?
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_3672

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????H@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
+__inference_SMILES_CNN_2_layer_call_fn_3781

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_26402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????H?2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????H@
 
_user_specified_nameinputs
?
V
*__inference_Concatenate_layer_call_fn_3794
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Concatenate_layer_call_and_return_conditional_losses_26972
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
{
&__inference_Dense_2_layer_call_fn_3908

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_2_layer_call_and_return_conditional_losses_28312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_CNN_FCNN_Model_layer_call_fn_3153
protein_input	
smiles_input	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprotein_inputsmiles_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_31102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????

'
_user_specified_nameProtein_Input:UQ
'
_output_shapes
:?????????H
&
_user_specified_nameSMILES_Input
?
?
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_3697

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????
@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
@
 
_user_specified_nameinputs
?
e
I__inference_Prot_Global_Max_layer_call_and_return_conditional_losses_2398

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
o
E__inference_Concatenate_layer_call_and_return_conditional_losses_2697

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_3587

inputs	
identity^
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????
2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:??????????
2	
one_hot?
GatherV2/indicesConst*
_output_shapes
:*
dtype0*e
value\BZ"P                        	   
                                 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2one_hot:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:??????????
2

GatherV2j
IdentityIdentityGatherV2:output:0*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
a
(__inference_Dropout_0_layer_call_fn_3836

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_0_layer_call_and_return_conditional_losses_27452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
{
&__inference_Dense_1_layer_call_fn_3861

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_1_layer_call_and_return_conditional_losses_27742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_2512

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????H@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
C
'__inference_Prot_Emb_layer_call_fn_3592

inputs	
identity?
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_24712
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
D
(__inference_Dropout_0_layer_call_fn_3841

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_0_layer_call_and_return_conditional_losses_27502
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_Dropout_1_layer_call_fn_3888

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_1_layer_call_and_return_conditional_losses_28072
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_3609

inputs	
identity]
CastCastinputs*

DstT0*

SrcT0	*'
_output_shapes
:?????????H2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????H2	
one_hot?
GatherV2/indicesConst*
_output_shapes
:*
dtype0*}
valuetBr"h                        	   
                                                   2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2one_hot:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????H2

GatherV2i
IdentityIdentityGatherV2:output:0*
T0*+
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????H:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
C
'__inference_Prot_Emb_layer_call_fn_3597

inputs	
identity?
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_24832
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_2576

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????H@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????H@
 
_user_specified_nameinputs
?
?
+__inference_SMILES_CNN_1_layer_call_fn_3731

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_25762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????H@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????H@
 
_user_specified_nameinputs
?	
?
A__inference_Dense_0_layer_call_and_return_conditional_losses_3805

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
C__inference_Dropout_0_layer_call_and_return_conditional_losses_2745

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_3918

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
y
$__inference_dense_layer_call_fn_3927

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_28572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_3722

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????H@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????H@
 
_user_specified_nameinputs
?
?
-__inference_CNN_FCNN_Model_layer_call_fn_3563
inputs_0	
inputs_1	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_31102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????H
"
_user_specified_name
inputs/1
?
{
&__inference_Dense_0_layer_call_fn_3814

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_0_layer_call_and_return_conditional_losses_27172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_3575

inputs	
identity^
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????
2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:??????????
2	
one_hot?
GatherV2/indicesConst*
_output_shapes
:*
dtype0*e
value\BZ"P                        	   
                                 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2one_hot:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:??????????
2

GatherV2j
IdentityIdentityGatherV2:output:0*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
??
?!
 __inference__traced_restore_4357
file_prefix&
"assignvariableop_prot_cnn_0_kernel&
"assignvariableop_1_prot_cnn_0_bias*
&assignvariableop_2_smiles_cnn_0_kernel(
$assignvariableop_3_smiles_cnn_0_bias(
$assignvariableop_4_prot_cnn_1_kernel&
"assignvariableop_5_prot_cnn_1_bias*
&assignvariableop_6_smiles_cnn_1_kernel(
$assignvariableop_7_smiles_cnn_1_bias(
$assignvariableop_8_prot_cnn_2_kernel&
"assignvariableop_9_prot_cnn_2_bias+
'assignvariableop_10_smiles_cnn_2_kernel)
%assignvariableop_11_smiles_cnn_2_bias&
"assignvariableop_12_dense_0_kernel$
 assignvariableop_13_dense_0_bias&
"assignvariableop_14_dense_1_kernel$
 assignvariableop_15_dense_1_bias&
"assignvariableop_16_dense_2_kernel$
 assignvariableop_17_dense_2_bias$
 assignvariableop_18_dense_kernel"
assignvariableop_19_dense_bias
assignvariableop_20_total
assignvariableop_21_count
assignvariableop_22_total_1
assignvariableop_23_count_1
assignvariableop_24_total_2
assignvariableop_25_count_2+
'assignvariableop_26_prot_cnn_0_kernel_m)
%assignvariableop_27_prot_cnn_0_bias_m-
)assignvariableop_28_smiles_cnn_0_kernel_m+
'assignvariableop_29_smiles_cnn_0_bias_m+
'assignvariableop_30_prot_cnn_1_kernel_m)
%assignvariableop_31_prot_cnn_1_bias_m-
)assignvariableop_32_smiles_cnn_1_kernel_m+
'assignvariableop_33_smiles_cnn_1_bias_m+
'assignvariableop_34_prot_cnn_2_kernel_m)
%assignvariableop_35_prot_cnn_2_bias_m-
)assignvariableop_36_smiles_cnn_2_kernel_m+
'assignvariableop_37_smiles_cnn_2_bias_m(
$assignvariableop_38_dense_0_kernel_m&
"assignvariableop_39_dense_0_bias_m(
$assignvariableop_40_dense_1_kernel_m&
"assignvariableop_41_dense_1_bias_m(
$assignvariableop_42_dense_2_kernel_m&
"assignvariableop_43_dense_2_bias_m&
"assignvariableop_44_dense_kernel_m$
 assignvariableop_45_dense_bias_m+
'assignvariableop_46_prot_cnn_0_kernel_v)
%assignvariableop_47_prot_cnn_0_bias_v-
)assignvariableop_48_smiles_cnn_0_kernel_v+
'assignvariableop_49_smiles_cnn_0_bias_v+
'assignvariableop_50_prot_cnn_1_kernel_v)
%assignvariableop_51_prot_cnn_1_bias_v-
)assignvariableop_52_smiles_cnn_1_kernel_v+
'assignvariableop_53_smiles_cnn_1_bias_v+
'assignvariableop_54_prot_cnn_2_kernel_v)
%assignvariableop_55_prot_cnn_2_bias_v-
)assignvariableop_56_smiles_cnn_2_kernel_v+
'assignvariableop_57_smiles_cnn_2_bias_v(
$assignvariableop_58_dense_0_kernel_v&
"assignvariableop_59_dense_0_bias_v(
$assignvariableop_60_dense_1_kernel_v&
"assignvariableop_61_dense_1_bias_v(
$assignvariableop_62_dense_2_kernel_v&
"assignvariableop_63_dense_2_bias_v&
"assignvariableop_64_dense_kernel_v$
 assignvariableop_65_dense_bias_v
identity_67??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?%
value?%B?%CB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?
value?B?CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_prot_cnn_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_prot_cnn_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_smiles_cnn_0_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_smiles_cnn_0_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_prot_cnn_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_prot_cnn_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_smiles_cnn_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_smiles_cnn_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_prot_cnn_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_prot_cnn_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_smiles_cnn_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_smiles_cnn_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_0_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_0_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_dense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_prot_cnn_0_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_prot_cnn_0_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_smiles_cnn_0_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_smiles_cnn_0_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_prot_cnn_1_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_prot_cnn_1_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_smiles_cnn_1_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_smiles_cnn_1_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_prot_cnn_2_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp%assignvariableop_35_prot_cnn_2_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_smiles_cnn_2_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_smiles_cnn_2_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp$assignvariableop_38_dense_0_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp"assignvariableop_39_dense_0_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp$assignvariableop_40_dense_1_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp"assignvariableop_41_dense_1_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp$assignvariableop_42_dense_2_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp"assignvariableop_43_dense_2_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_prot_cnn_0_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp%assignvariableop_47_prot_cnn_0_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_smiles_cnn_0_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp'assignvariableop_49_smiles_cnn_0_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp'assignvariableop_50_prot_cnn_1_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp%assignvariableop_51_prot_cnn_1_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_smiles_cnn_1_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp'assignvariableop_53_smiles_cnn_1_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp'assignvariableop_54_prot_cnn_2_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp%assignvariableop_55_prot_cnn_2_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_smiles_cnn_2_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp'assignvariableop_57_smiles_cnn_2_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp$assignvariableop_58_dense_0_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp"assignvariableop_59_dense_0_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp$assignvariableop_60_dense_1_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp"assignvariableop_61_dense_1_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp$assignvariableop_62_dense_2_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp"assignvariableop_63_dense_2_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp"assignvariableop_64_dense_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp assignvariableop_65_dense_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_659
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_66?
Identity_67IdentityIdentity_66:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_67"#
identity_67Identity_67:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
`
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_3621

inputs	
identity]
CastCastinputs*

DstT0*

SrcT0	*'
_output_shapes
:?????????H2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????H2	
one_hot?
GatherV2/indicesConst*
_output_shapes
:*
dtype0*}
valuetBr"h                        	   
                                                   2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2one_hot:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????H2

GatherV2i
IdentityIdentityGatherV2:output:0*
T0*+
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????H:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
J
.__inference_Prot_Global_Max_layer_call_fn_2404

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Prot_Global_Max_layer_call_and_return_conditional_losses_23982
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
a
(__inference_Dropout_1_layer_call_fn_3883

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_1_layer_call_and_return_conditional_losses_28022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_Dropout_1_layer_call_and_return_conditional_losses_2807

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_2471

inputs	
identity^
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????
2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:??????????
2	
one_hot?
GatherV2/indicesConst*
_output_shapes
:*
dtype0*e
value\BZ"P                        	   
                                 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2one_hot:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:??????????
2

GatherV2j
IdentityIdentityGatherV2:output:0*
T0*,
_output_shapes
:??????????
2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_2608

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????
@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
@
 
_user_specified_nameinputs
?
?
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_2672

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:??????????
?*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:??????????
?*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????
?2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:??????????
?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:??????????
?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
@
 
_user_specified_nameinputs
?
a
C__inference_Dropout_1_layer_call_and_return_conditional_losses_3878

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?K
?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_3110

inputs	
inputs_1	
smiles_cnn_0_3054
smiles_cnn_0_3056
prot_cnn_0_3059
prot_cnn_0_3061
smiles_cnn_1_3064
smiles_cnn_1_3066
prot_cnn_1_3069
prot_cnn_1_3071
smiles_cnn_2_3074
smiles_cnn_2_3076
prot_cnn_2_3079
prot_cnn_2_3081
dense_0_3087
dense_0_3089
dense_1_3093
dense_1_3095
dense_2_3099
dense_2_3101

dense_3104

dense_3106
identity??Dense_0/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?"Prot_CNN_0/StatefulPartitionedCall?"Prot_CNN_1/StatefulPartitionedCall?"Prot_CNN_2/StatefulPartitionedCall?$SMILES_CNN_0/StatefulPartitionedCall?$SMILES_CNN_1/StatefulPartitionedCall?$SMILES_CNN_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
SMILES_Emb/PartitionedCallPartitionedCallinputs_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_24462
SMILES_Emb/PartitionedCall?
Prot_Emb/PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_24832
Prot_Emb/PartitionedCall?
$SMILES_CNN_0/StatefulPartitionedCallStatefulPartitionedCall#SMILES_Emb/PartitionedCall:output:0smiles_cnn_0_3054smiles_cnn_0_3056*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_25122&
$SMILES_CNN_0/StatefulPartitionedCall?
"Prot_CNN_0/StatefulPartitionedCallStatefulPartitionedCall!Prot_Emb/PartitionedCall:output:0prot_cnn_0_3059prot_cnn_0_3061*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_25442$
"Prot_CNN_0/StatefulPartitionedCall?
$SMILES_CNN_1/StatefulPartitionedCallStatefulPartitionedCall-SMILES_CNN_0/StatefulPartitionedCall:output:0smiles_cnn_1_3064smiles_cnn_1_3066*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_25762&
$SMILES_CNN_1/StatefulPartitionedCall?
"Prot_CNN_1/StatefulPartitionedCallStatefulPartitionedCall+Prot_CNN_0/StatefulPartitionedCall:output:0prot_cnn_1_3069prot_cnn_1_3071*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_26082$
"Prot_CNN_1/StatefulPartitionedCall?
$SMILES_CNN_2/StatefulPartitionedCallStatefulPartitionedCall-SMILES_CNN_1/StatefulPartitionedCall:output:0smiles_cnn_2_3074smiles_cnn_2_3076*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_26402&
$SMILES_CNN_2/StatefulPartitionedCall?
"Prot_CNN_2/StatefulPartitionedCallStatefulPartitionedCall+Prot_CNN_1/StatefulPartitionedCall:output:0prot_cnn_2_3079prot_cnn_2_3081*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:??????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_26722$
"Prot_CNN_2/StatefulPartitionedCall?
Prot_Global_Max/PartitionedCallPartitionedCall+Prot_CNN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Prot_Global_Max_layer_call_and_return_conditional_losses_23982!
Prot_Global_Max/PartitionedCall?
!SMILES_Global_Max/PartitionedCallPartitionedCall-SMILES_CNN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_SMILES_Global_Max_layer_call_and_return_conditional_losses_24112#
!SMILES_Global_Max/PartitionedCall?
Concatenate/PartitionedCallPartitionedCall(Prot_Global_Max/PartitionedCall:output:0*SMILES_Global_Max/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Concatenate_layer_call_and_return_conditional_losses_26972
Concatenate/PartitionedCall?
Dense_0/StatefulPartitionedCallStatefulPartitionedCall$Concatenate/PartitionedCall:output:0dense_0_3087dense_0_3089*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_0_layer_call_and_return_conditional_losses_27172!
Dense_0/StatefulPartitionedCall?
Dropout_0/PartitionedCallPartitionedCall(Dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_0_layer_call_and_return_conditional_losses_27502
Dropout_0/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall"Dropout_0/PartitionedCall:output:0dense_1_3093dense_1_3095*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_1_layer_call_and_return_conditional_losses_27742!
Dense_1/StatefulPartitionedCall?
Dropout_1/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_1_layer_call_and_return_conditional_losses_28072
Dropout_1/PartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall"Dropout_1/PartitionedCall:output:0dense_2_3099dense_2_3101*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_2_layer_call_and_return_conditional_losses_28312!
Dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0
dense_3104
dense_3106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_28572
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0 ^Dense_0/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall#^Prot_CNN_0/StatefulPartitionedCall#^Prot_CNN_1/StatefulPartitionedCall#^Prot_CNN_2/StatefulPartitionedCall%^SMILES_CNN_0/StatefulPartitionedCall%^SMILES_CNN_1/StatefulPartitionedCall%^SMILES_CNN_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::2B
Dense_0/StatefulPartitionedCallDense_0/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2H
"Prot_CNN_0/StatefulPartitionedCall"Prot_CNN_0/StatefulPartitionedCall2H
"Prot_CNN_1/StatefulPartitionedCall"Prot_CNN_1/StatefulPartitionedCall2H
"Prot_CNN_2/StatefulPartitionedCall"Prot_CNN_2/StatefulPartitionedCall2L
$SMILES_CNN_0/StatefulPartitionedCall$SMILES_CNN_0/StatefulPartitionedCall2L
$SMILES_CNN_1/StatefulPartitionedCall$SMILES_CNN_1/StatefulPartitionedCall2L
$SMILES_CNN_2/StatefulPartitionedCall$SMILES_CNN_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
??
?
__inference__traced_save_4149
file_prefix0
,savev2_prot_cnn_0_kernel_read_readvariableop.
*savev2_prot_cnn_0_bias_read_readvariableop2
.savev2_smiles_cnn_0_kernel_read_readvariableop0
,savev2_smiles_cnn_0_bias_read_readvariableop0
,savev2_prot_cnn_1_kernel_read_readvariableop.
*savev2_prot_cnn_1_bias_read_readvariableop2
.savev2_smiles_cnn_1_kernel_read_readvariableop0
,savev2_smiles_cnn_1_bias_read_readvariableop0
,savev2_prot_cnn_2_kernel_read_readvariableop.
*savev2_prot_cnn_2_bias_read_readvariableop2
.savev2_smiles_cnn_2_kernel_read_readvariableop0
,savev2_smiles_cnn_2_bias_read_readvariableop-
)savev2_dense_0_kernel_read_readvariableop+
'savev2_dense_0_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop2
.savev2_prot_cnn_0_kernel_m_read_readvariableop0
,savev2_prot_cnn_0_bias_m_read_readvariableop4
0savev2_smiles_cnn_0_kernel_m_read_readvariableop2
.savev2_smiles_cnn_0_bias_m_read_readvariableop2
.savev2_prot_cnn_1_kernel_m_read_readvariableop0
,savev2_prot_cnn_1_bias_m_read_readvariableop4
0savev2_smiles_cnn_1_kernel_m_read_readvariableop2
.savev2_smiles_cnn_1_bias_m_read_readvariableop2
.savev2_prot_cnn_2_kernel_m_read_readvariableop0
,savev2_prot_cnn_2_bias_m_read_readvariableop4
0savev2_smiles_cnn_2_kernel_m_read_readvariableop2
.savev2_smiles_cnn_2_bias_m_read_readvariableop/
+savev2_dense_0_kernel_m_read_readvariableop-
)savev2_dense_0_bias_m_read_readvariableop/
+savev2_dense_1_kernel_m_read_readvariableop-
)savev2_dense_1_bias_m_read_readvariableop/
+savev2_dense_2_kernel_m_read_readvariableop-
)savev2_dense_2_bias_m_read_readvariableop-
)savev2_dense_kernel_m_read_readvariableop+
'savev2_dense_bias_m_read_readvariableop2
.savev2_prot_cnn_0_kernel_v_read_readvariableop0
,savev2_prot_cnn_0_bias_v_read_readvariableop4
0savev2_smiles_cnn_0_kernel_v_read_readvariableop2
.savev2_smiles_cnn_0_bias_v_read_readvariableop2
.savev2_prot_cnn_1_kernel_v_read_readvariableop0
,savev2_prot_cnn_1_bias_v_read_readvariableop4
0savev2_smiles_cnn_1_kernel_v_read_readvariableop2
.savev2_smiles_cnn_1_bias_v_read_readvariableop2
.savev2_prot_cnn_2_kernel_v_read_readvariableop0
,savev2_prot_cnn_2_bias_v_read_readvariableop4
0savev2_smiles_cnn_2_kernel_v_read_readvariableop2
.savev2_smiles_cnn_2_bias_v_read_readvariableop/
+savev2_dense_0_kernel_v_read_readvariableop-
)savev2_dense_0_bias_v_read_readvariableop/
+savev2_dense_1_kernel_v_read_readvariableop-
)savev2_dense_1_bias_v_read_readvariableop/
+savev2_dense_2_kernel_v_read_readvariableop-
)savev2_dense_2_bias_v_read_readvariableop-
)savev2_dense_kernel_v_read_readvariableop+
'savev2_dense_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?%
value?%B?%CB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?
value?B?CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_prot_cnn_0_kernel_read_readvariableop*savev2_prot_cnn_0_bias_read_readvariableop.savev2_smiles_cnn_0_kernel_read_readvariableop,savev2_smiles_cnn_0_bias_read_readvariableop,savev2_prot_cnn_1_kernel_read_readvariableop*savev2_prot_cnn_1_bias_read_readvariableop.savev2_smiles_cnn_1_kernel_read_readvariableop,savev2_smiles_cnn_1_bias_read_readvariableop,savev2_prot_cnn_2_kernel_read_readvariableop*savev2_prot_cnn_2_bias_read_readvariableop.savev2_smiles_cnn_2_kernel_read_readvariableop,savev2_smiles_cnn_2_bias_read_readvariableop)savev2_dense_0_kernel_read_readvariableop'savev2_dense_0_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop.savev2_prot_cnn_0_kernel_m_read_readvariableop,savev2_prot_cnn_0_bias_m_read_readvariableop0savev2_smiles_cnn_0_kernel_m_read_readvariableop.savev2_smiles_cnn_0_bias_m_read_readvariableop.savev2_prot_cnn_1_kernel_m_read_readvariableop,savev2_prot_cnn_1_bias_m_read_readvariableop0savev2_smiles_cnn_1_kernel_m_read_readvariableop.savev2_smiles_cnn_1_bias_m_read_readvariableop.savev2_prot_cnn_2_kernel_m_read_readvariableop,savev2_prot_cnn_2_bias_m_read_readvariableop0savev2_smiles_cnn_2_kernel_m_read_readvariableop.savev2_smiles_cnn_2_bias_m_read_readvariableop+savev2_dense_0_kernel_m_read_readvariableop)savev2_dense_0_bias_m_read_readvariableop+savev2_dense_1_kernel_m_read_readvariableop)savev2_dense_1_bias_m_read_readvariableop+savev2_dense_2_kernel_m_read_readvariableop)savev2_dense_2_bias_m_read_readvariableop)savev2_dense_kernel_m_read_readvariableop'savev2_dense_bias_m_read_readvariableop.savev2_prot_cnn_0_kernel_v_read_readvariableop,savev2_prot_cnn_0_bias_v_read_readvariableop0savev2_smiles_cnn_0_kernel_v_read_readvariableop.savev2_smiles_cnn_0_bias_v_read_readvariableop.savev2_prot_cnn_1_kernel_v_read_readvariableop,savev2_prot_cnn_1_bias_v_read_readvariableop0savev2_smiles_cnn_1_kernel_v_read_readvariableop.savev2_smiles_cnn_1_bias_v_read_readvariableop.savev2_prot_cnn_2_kernel_v_read_readvariableop,savev2_prot_cnn_2_bias_v_read_readvariableop0savev2_smiles_cnn_2_kernel_v_read_readvariableop.savev2_smiles_cnn_2_bias_v_read_readvariableop+savev2_dense_0_kernel_v_read_readvariableop)savev2_dense_0_bias_v_read_readvariableop+savev2_dense_1_kernel_v_read_readvariableop)savev2_dense_1_bias_v_read_readvariableop+savev2_dense_2_kernel_v_read_readvariableop)savev2_dense_2_bias_v_read_readvariableop)savev2_dense_kernel_v_read_readvariableop'savev2_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Q
dtypesG
E2C2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@:@:@@:@:@@:@:@?:?:@?:?:
??:?:
??:?:
??:?:	?:: : : : : : :@:@:@:@:@@:@:@@:@:@?:?:@?:?:
??:?:
??:?:
??:?:	?::@:@:@:@:@@:@:@@:@:@?:?:@?:?:
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:)	%
#
_output_shapes
:@?:!


_output_shapes	
:?:)%
#
_output_shapes
:@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@:  

_output_shapes
:@:(!$
"
_output_shapes
:@@: "

_output_shapes
:@:)#%
#
_output_shapes
:@?:!$

_output_shapes	
:?:)%%
#
_output_shapes
:@?:!&

_output_shapes	
:?:&'"
 
_output_shapes
:
??:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:%-!

_output_shapes
:	?: .

_output_shapes
::(/$
"
_output_shapes
:@: 0

_output_shapes
:@:(1$
"
_output_shapes
:@: 2

_output_shapes
:@:(3$
"
_output_shapes
:@@: 4

_output_shapes
:@:(5$
"
_output_shapes
:@@: 6

_output_shapes
:@:)7%
#
_output_shapes
:@?:!8

_output_shapes	
:?:)9%
#
_output_shapes
:@?:!:

_output_shapes	
:?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:&?"
 
_output_shapes
:
??:!@

_output_shapes	
:?:%A!

_output_shapes
:	?: B

_output_shapes
::C

_output_shapes
: 
?
E
)__inference_SMILES_Emb_layer_call_fn_3631

inputs	
identity?
PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_24462
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????H:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
+__inference_SMILES_CNN_0_layer_call_fn_3681

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_25122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????H@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????H::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????H
 
_user_specified_nameinputs
?	
?
A__inference_Dense_0_layer_call_and_return_conditional_losses_2717

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_3747

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:??????????
?*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:??????????
?*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????
?2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:??????????
?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:??????????
?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
@
 
_user_specified_nameinputs
?
~
)__inference_Prot_CNN_2_layer_call_fn_3756

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:??????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_26722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:??????????
?2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
@
 
_user_specified_nameinputs
?
b
C__inference_Dropout_0_layer_call_and_return_conditional_losses_3826

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?N
?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_3002

inputs	
inputs_1	
smiles_cnn_0_2946
smiles_cnn_0_2948
prot_cnn_0_2951
prot_cnn_0_2953
smiles_cnn_1_2956
smiles_cnn_1_2958
prot_cnn_1_2961
prot_cnn_1_2963
smiles_cnn_2_2966
smiles_cnn_2_2968
prot_cnn_2_2971
prot_cnn_2_2973
dense_0_2979
dense_0_2981
dense_1_2985
dense_1_2987
dense_2_2991
dense_2_2993

dense_2996

dense_2998
identity??Dense_0/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?!Dropout_0/StatefulPartitionedCall?!Dropout_1/StatefulPartitionedCall?"Prot_CNN_0/StatefulPartitionedCall?"Prot_CNN_1/StatefulPartitionedCall?"Prot_CNN_2/StatefulPartitionedCall?$SMILES_CNN_0/StatefulPartitionedCall?$SMILES_CNN_1/StatefulPartitionedCall?$SMILES_CNN_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
SMILES_Emb/PartitionedCallPartitionedCallinputs_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_24342
SMILES_Emb/PartitionedCall?
Prot_Emb/PartitionedCallPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_24712
Prot_Emb/PartitionedCall?
$SMILES_CNN_0/StatefulPartitionedCallStatefulPartitionedCall#SMILES_Emb/PartitionedCall:output:0smiles_cnn_0_2946smiles_cnn_0_2948*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_25122&
$SMILES_CNN_0/StatefulPartitionedCall?
"Prot_CNN_0/StatefulPartitionedCallStatefulPartitionedCall!Prot_Emb/PartitionedCall:output:0prot_cnn_0_2951prot_cnn_0_2953*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_25442$
"Prot_CNN_0/StatefulPartitionedCall?
$SMILES_CNN_1/StatefulPartitionedCallStatefulPartitionedCall-SMILES_CNN_0/StatefulPartitionedCall:output:0smiles_cnn_1_2956smiles_cnn_1_2958*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_25762&
$SMILES_CNN_1/StatefulPartitionedCall?
"Prot_CNN_1/StatefulPartitionedCallStatefulPartitionedCall+Prot_CNN_0/StatefulPartitionedCall:output:0prot_cnn_1_2961prot_cnn_1_2963*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_26082$
"Prot_CNN_1/StatefulPartitionedCall?
$SMILES_CNN_2/StatefulPartitionedCallStatefulPartitionedCall-SMILES_CNN_1/StatefulPartitionedCall:output:0smiles_cnn_2_2966smiles_cnn_2_2968*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_26402&
$SMILES_CNN_2/StatefulPartitionedCall?
"Prot_CNN_2/StatefulPartitionedCallStatefulPartitionedCall+Prot_CNN_1/StatefulPartitionedCall:output:0prot_cnn_2_2971prot_cnn_2_2973*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:??????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_26722$
"Prot_CNN_2/StatefulPartitionedCall?
Prot_Global_Max/PartitionedCallPartitionedCall+Prot_CNN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Prot_Global_Max_layer_call_and_return_conditional_losses_23982!
Prot_Global_Max/PartitionedCall?
!SMILES_Global_Max/PartitionedCallPartitionedCall-SMILES_CNN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_SMILES_Global_Max_layer_call_and_return_conditional_losses_24112#
!SMILES_Global_Max/PartitionedCall?
Concatenate/PartitionedCallPartitionedCall(Prot_Global_Max/PartitionedCall:output:0*SMILES_Global_Max/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Concatenate_layer_call_and_return_conditional_losses_26972
Concatenate/PartitionedCall?
Dense_0/StatefulPartitionedCallStatefulPartitionedCall$Concatenate/PartitionedCall:output:0dense_0_2979dense_0_2981*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_0_layer_call_and_return_conditional_losses_27172!
Dense_0/StatefulPartitionedCall?
!Dropout_0/StatefulPartitionedCallStatefulPartitionedCall(Dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_0_layer_call_and_return_conditional_losses_27452#
!Dropout_0/StatefulPartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall*Dropout_0/StatefulPartitionedCall:output:0dense_1_2985dense_1_2987*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_1_layer_call_and_return_conditional_losses_27742!
Dense_1/StatefulPartitionedCall?
!Dropout_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0"^Dropout_0/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_1_layer_call_and_return_conditional_losses_28022#
!Dropout_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall*Dropout_1/StatefulPartitionedCall:output:0dense_2_2991dense_2_2993*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_2_layer_call_and_return_conditional_losses_28312!
Dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0
dense_2996
dense_2998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_28572
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0 ^Dense_0/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall"^Dropout_0/StatefulPartitionedCall"^Dropout_1/StatefulPartitionedCall#^Prot_CNN_0/StatefulPartitionedCall#^Prot_CNN_1/StatefulPartitionedCall#^Prot_CNN_2/StatefulPartitionedCall%^SMILES_CNN_0/StatefulPartitionedCall%^SMILES_CNN_1/StatefulPartitionedCall%^SMILES_CNN_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::2B
Dense_0/StatefulPartitionedCallDense_0/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2F
!Dropout_0/StatefulPartitionedCall!Dropout_0/StatefulPartitionedCall2F
!Dropout_1/StatefulPartitionedCall!Dropout_1/StatefulPartitionedCall2H
"Prot_CNN_0/StatefulPartitionedCall"Prot_CNN_0/StatefulPartitionedCall2H
"Prot_CNN_1/StatefulPartitionedCall"Prot_CNN_1/StatefulPartitionedCall2H
"Prot_CNN_2/StatefulPartitionedCall"Prot_CNN_2/StatefulPartitionedCall2L
$SMILES_CNN_0/StatefulPartitionedCall$SMILES_CNN_0/StatefulPartitionedCall2L
$SMILES_CNN_1/StatefulPartitionedCall$SMILES_CNN_1/StatefulPartitionedCall2L
$SMILES_CNN_2/StatefulPartitionedCall$SMILES_CNN_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?K
?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_2936
protein_input	
smiles_input	
smiles_cnn_0_2880
smiles_cnn_0_2882
prot_cnn_0_2885
prot_cnn_0_2887
smiles_cnn_1_2890
smiles_cnn_1_2892
prot_cnn_1_2895
prot_cnn_1_2897
smiles_cnn_2_2900
smiles_cnn_2_2902
prot_cnn_2_2905
prot_cnn_2_2907
dense_0_2913
dense_0_2915
dense_1_2919
dense_1_2921
dense_2_2925
dense_2_2927

dense_2930

dense_2932
identity??Dense_0/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?"Prot_CNN_0/StatefulPartitionedCall?"Prot_CNN_1/StatefulPartitionedCall?"Prot_CNN_2/StatefulPartitionedCall?$SMILES_CNN_0/StatefulPartitionedCall?$SMILES_CNN_1/StatefulPartitionedCall?$SMILES_CNN_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
SMILES_Emb/PartitionedCallPartitionedCallsmiles_input*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_24462
SMILES_Emb/PartitionedCall?
Prot_Emb/PartitionedCallPartitionedCallprotein_input*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_24832
Prot_Emb/PartitionedCall?
$SMILES_CNN_0/StatefulPartitionedCallStatefulPartitionedCall#SMILES_Emb/PartitionedCall:output:0smiles_cnn_0_2880smiles_cnn_0_2882*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_25122&
$SMILES_CNN_0/StatefulPartitionedCall?
"Prot_CNN_0/StatefulPartitionedCallStatefulPartitionedCall!Prot_Emb/PartitionedCall:output:0prot_cnn_0_2885prot_cnn_0_2887*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_25442$
"Prot_CNN_0/StatefulPartitionedCall?
$SMILES_CNN_1/StatefulPartitionedCallStatefulPartitionedCall-SMILES_CNN_0/StatefulPartitionedCall:output:0smiles_cnn_1_2890smiles_cnn_1_2892*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_25762&
$SMILES_CNN_1/StatefulPartitionedCall?
"Prot_CNN_1/StatefulPartitionedCallStatefulPartitionedCall+Prot_CNN_0/StatefulPartitionedCall:output:0prot_cnn_1_2895prot_cnn_1_2897*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_26082$
"Prot_CNN_1/StatefulPartitionedCall?
$SMILES_CNN_2/StatefulPartitionedCallStatefulPartitionedCall-SMILES_CNN_1/StatefulPartitionedCall:output:0smiles_cnn_2_2900smiles_cnn_2_2902*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_26402&
$SMILES_CNN_2/StatefulPartitionedCall?
"Prot_CNN_2/StatefulPartitionedCallStatefulPartitionedCall+Prot_CNN_1/StatefulPartitionedCall:output:0prot_cnn_2_2905prot_cnn_2_2907*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:??????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_26722$
"Prot_CNN_2/StatefulPartitionedCall?
Prot_Global_Max/PartitionedCallPartitionedCall+Prot_CNN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Prot_Global_Max_layer_call_and_return_conditional_losses_23982!
Prot_Global_Max/PartitionedCall?
!SMILES_Global_Max/PartitionedCallPartitionedCall-SMILES_CNN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_SMILES_Global_Max_layer_call_and_return_conditional_losses_24112#
!SMILES_Global_Max/PartitionedCall?
Concatenate/PartitionedCallPartitionedCall(Prot_Global_Max/PartitionedCall:output:0*SMILES_Global_Max/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Concatenate_layer_call_and_return_conditional_losses_26972
Concatenate/PartitionedCall?
Dense_0/StatefulPartitionedCallStatefulPartitionedCall$Concatenate/PartitionedCall:output:0dense_0_2913dense_0_2915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_0_layer_call_and_return_conditional_losses_27172!
Dense_0/StatefulPartitionedCall?
Dropout_0/PartitionedCallPartitionedCall(Dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_0_layer_call_and_return_conditional_losses_27502
Dropout_0/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall"Dropout_0/PartitionedCall:output:0dense_1_2919dense_1_2921*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_1_layer_call_and_return_conditional_losses_27742!
Dense_1/StatefulPartitionedCall?
Dropout_1/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_1_layer_call_and_return_conditional_losses_28072
Dropout_1/PartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall"Dropout_1/PartitionedCall:output:0dense_2_2925dense_2_2927*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_2_layer_call_and_return_conditional_losses_28312!
Dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0
dense_2930
dense_2932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_28572
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0 ^Dense_0/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall#^Prot_CNN_0/StatefulPartitionedCall#^Prot_CNN_1/StatefulPartitionedCall#^Prot_CNN_2/StatefulPartitionedCall%^SMILES_CNN_0/StatefulPartitionedCall%^SMILES_CNN_1/StatefulPartitionedCall%^SMILES_CNN_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::2B
Dense_0/StatefulPartitionedCallDense_0/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2H
"Prot_CNN_0/StatefulPartitionedCall"Prot_CNN_0/StatefulPartitionedCall2H
"Prot_CNN_1/StatefulPartitionedCall"Prot_CNN_1/StatefulPartitionedCall2H
"Prot_CNN_2/StatefulPartitionedCall"Prot_CNN_2/StatefulPartitionedCall2L
$SMILES_CNN_0/StatefulPartitionedCall$SMILES_CNN_0/StatefulPartitionedCall2L
$SMILES_CNN_1/StatefulPartitionedCall$SMILES_CNN_1/StatefulPartitionedCall2L
$SMILES_CNN_2/StatefulPartitionedCall$SMILES_CNN_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
(
_output_shapes
:??????????

'
_user_specified_nameProtein_Input:UQ
'
_output_shapes
:?????????H
&
_user_specified_nameSMILES_Input
?
`
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_2434

inputs	
identity]
CastCastinputs*

DstT0*

SrcT0	*'
_output_shapes
:?????????H2
Casti
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_value`
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
one_hot/depth?
one_hotOneHotCast:y:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????H2	
one_hot?
GatherV2/indicesConst*
_output_shapes
:*
dtype0*}
valuetBr"h                        	   
                                                   2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2/axis?
GatherV2GatherV2one_hot:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????H2

GatherV2i
IdentityIdentityGatherV2:output:0*
T0*+
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????H:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?N
?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_2874
protein_input	
smiles_input	
smiles_cnn_0_2523
smiles_cnn_0_2525
prot_cnn_0_2555
prot_cnn_0_2557
smiles_cnn_1_2587
smiles_cnn_1_2589
prot_cnn_1_2619
prot_cnn_1_2621
smiles_cnn_2_2651
smiles_cnn_2_2653
prot_cnn_2_2683
prot_cnn_2_2685
dense_0_2728
dense_0_2730
dense_1_2785
dense_1_2787
dense_2_2842
dense_2_2844

dense_2868

dense_2870
identity??Dense_0/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?!Dropout_0/StatefulPartitionedCall?!Dropout_1/StatefulPartitionedCall?"Prot_CNN_0/StatefulPartitionedCall?"Prot_CNN_1/StatefulPartitionedCall?"Prot_CNN_2/StatefulPartitionedCall?$SMILES_CNN_0/StatefulPartitionedCall?$SMILES_CNN_1/StatefulPartitionedCall?$SMILES_CNN_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
SMILES_Emb/PartitionedCallPartitionedCallsmiles_input*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_24342
SMILES_Emb/PartitionedCall?
Prot_Emb/PartitionedCallPartitionedCallprotein_input*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_24712
Prot_Emb/PartitionedCall?
$SMILES_CNN_0/StatefulPartitionedCallStatefulPartitionedCall#SMILES_Emb/PartitionedCall:output:0smiles_cnn_0_2523smiles_cnn_0_2525*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_25122&
$SMILES_CNN_0/StatefulPartitionedCall?
"Prot_CNN_0/StatefulPartitionedCallStatefulPartitionedCall!Prot_Emb/PartitionedCall:output:0prot_cnn_0_2555prot_cnn_0_2557*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_25442$
"Prot_CNN_0/StatefulPartitionedCall?
$SMILES_CNN_1/StatefulPartitionedCallStatefulPartitionedCall-SMILES_CNN_0/StatefulPartitionedCall:output:0smiles_cnn_1_2587smiles_cnn_1_2589*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????H@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_25762&
$SMILES_CNN_1/StatefulPartitionedCall?
"Prot_CNN_1/StatefulPartitionedCallStatefulPartitionedCall+Prot_CNN_0/StatefulPartitionedCall:output:0prot_cnn_1_2619prot_cnn_1_2621*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_26082$
"Prot_CNN_1/StatefulPartitionedCall?
$SMILES_CNN_2/StatefulPartitionedCallStatefulPartitionedCall-SMILES_CNN_1/StatefulPartitionedCall:output:0smiles_cnn_2_2651smiles_cnn_2_2653*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????H?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_26402&
$SMILES_CNN_2/StatefulPartitionedCall?
"Prot_CNN_2/StatefulPartitionedCallStatefulPartitionedCall+Prot_CNN_1/StatefulPartitionedCall:output:0prot_cnn_2_2683prot_cnn_2_2685*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:??????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_26722$
"Prot_CNN_2/StatefulPartitionedCall?
Prot_Global_Max/PartitionedCallPartitionedCall+Prot_CNN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Prot_Global_Max_layer_call_and_return_conditional_losses_23982!
Prot_Global_Max/PartitionedCall?
!SMILES_Global_Max/PartitionedCallPartitionedCall-SMILES_CNN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_SMILES_Global_Max_layer_call_and_return_conditional_losses_24112#
!SMILES_Global_Max/PartitionedCall?
Concatenate/PartitionedCallPartitionedCall(Prot_Global_Max/PartitionedCall:output:0*SMILES_Global_Max/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Concatenate_layer_call_and_return_conditional_losses_26972
Concatenate/PartitionedCall?
Dense_0/StatefulPartitionedCallStatefulPartitionedCall$Concatenate/PartitionedCall:output:0dense_0_2728dense_0_2730*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_0_layer_call_and_return_conditional_losses_27172!
Dense_0/StatefulPartitionedCall?
!Dropout_0/StatefulPartitionedCallStatefulPartitionedCall(Dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_0_layer_call_and_return_conditional_losses_27452#
!Dropout_0/StatefulPartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall*Dropout_0/StatefulPartitionedCall:output:0dense_1_2785dense_1_2787*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_1_layer_call_and_return_conditional_losses_27742!
Dense_1/StatefulPartitionedCall?
!Dropout_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0"^Dropout_0/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Dropout_1_layer_call_and_return_conditional_losses_28022#
!Dropout_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall*Dropout_1/StatefulPartitionedCall:output:0dense_2_2842dense_2_2844*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Dense_2_layer_call_and_return_conditional_losses_28312!
Dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0
dense_2868
dense_2870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_28572
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0 ^Dense_0/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall"^Dropout_0/StatefulPartitionedCall"^Dropout_1/StatefulPartitionedCall#^Prot_CNN_0/StatefulPartitionedCall#^Prot_CNN_1/StatefulPartitionedCall#^Prot_CNN_2/StatefulPartitionedCall%^SMILES_CNN_0/StatefulPartitionedCall%^SMILES_CNN_1/StatefulPartitionedCall%^SMILES_CNN_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::2B
Dense_0/StatefulPartitionedCallDense_0/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2F
!Dropout_0/StatefulPartitionedCall!Dropout_0/StatefulPartitionedCall2F
!Dropout_1/StatefulPartitionedCall!Dropout_1/StatefulPartitionedCall2H
"Prot_CNN_0/StatefulPartitionedCall"Prot_CNN_0/StatefulPartitionedCall2H
"Prot_CNN_1/StatefulPartitionedCall"Prot_CNN_1/StatefulPartitionedCall2H
"Prot_CNN_2/StatefulPartitionedCall"Prot_CNN_2/StatefulPartitionedCall2L
$SMILES_CNN_0/StatefulPartitionedCall$SMILES_CNN_0/StatefulPartitionedCall2L
$SMILES_CNN_1/StatefulPartitionedCall$SMILES_CNN_1/StatefulPartitionedCall2L
$SMILES_CNN_2/StatefulPartitionedCall$SMILES_CNN_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
(
_output_shapes
:??????????

'
_user_specified_nameProtein_Input:UQ
'
_output_shapes
:?????????H
&
_user_specified_nameSMILES_Input
?	
?
A__inference_Dense_1_layer_call_and_return_conditional_losses_2774

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_2544

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????
@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
??
?
__inference__wrapped_model_2391
protein_input	
smiles_input	K
Gcnn_fcnn_model_smiles_cnn_0_conv1d_expanddims_1_readvariableop_resource?
;cnn_fcnn_model_smiles_cnn_0_biasadd_readvariableop_resourceI
Ecnn_fcnn_model_prot_cnn_0_conv1d_expanddims_1_readvariableop_resource=
9cnn_fcnn_model_prot_cnn_0_biasadd_readvariableop_resourceK
Gcnn_fcnn_model_smiles_cnn_1_conv1d_expanddims_1_readvariableop_resource?
;cnn_fcnn_model_smiles_cnn_1_biasadd_readvariableop_resourceI
Ecnn_fcnn_model_prot_cnn_1_conv1d_expanddims_1_readvariableop_resource=
9cnn_fcnn_model_prot_cnn_1_biasadd_readvariableop_resourceK
Gcnn_fcnn_model_smiles_cnn_2_conv1d_expanddims_1_readvariableop_resource?
;cnn_fcnn_model_smiles_cnn_2_biasadd_readvariableop_resourceI
Ecnn_fcnn_model_prot_cnn_2_conv1d_expanddims_1_readvariableop_resource=
9cnn_fcnn_model_prot_cnn_2_biasadd_readvariableop_resource9
5cnn_fcnn_model_dense_0_matmul_readvariableop_resource:
6cnn_fcnn_model_dense_0_biasadd_readvariableop_resource9
5cnn_fcnn_model_dense_1_matmul_readvariableop_resource:
6cnn_fcnn_model_dense_1_biasadd_readvariableop_resource9
5cnn_fcnn_model_dense_2_matmul_readvariableop_resource:
6cnn_fcnn_model_dense_2_biasadd_readvariableop_resource7
3cnn_fcnn_model_dense_matmul_readvariableop_resource8
4cnn_fcnn_model_dense_biasadd_readvariableop_resource
identity??-CNN_FCNN_Model/Dense_0/BiasAdd/ReadVariableOp?,CNN_FCNN_Model/Dense_0/MatMul/ReadVariableOp?-CNN_FCNN_Model/Dense_1/BiasAdd/ReadVariableOp?,CNN_FCNN_Model/Dense_1/MatMul/ReadVariableOp?-CNN_FCNN_Model/Dense_2/BiasAdd/ReadVariableOp?,CNN_FCNN_Model/Dense_2/MatMul/ReadVariableOp?0CNN_FCNN_Model/Prot_CNN_0/BiasAdd/ReadVariableOp?<CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?0CNN_FCNN_Model/Prot_CNN_1/BiasAdd/ReadVariableOp?<CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?0CNN_FCNN_Model/Prot_CNN_2/BiasAdd/ReadVariableOp?<CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?2CNN_FCNN_Model/SMILES_CNN_0/BiasAdd/ReadVariableOp?>CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?2CNN_FCNN_Model/SMILES_CNN_1/BiasAdd/ReadVariableOp?>CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?2CNN_FCNN_Model/SMILES_CNN_2/BiasAdd/ReadVariableOp?>CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?+CNN_FCNN_Model/dense/BiasAdd/ReadVariableOp?*CNN_FCNN_Model/dense/MatMul/ReadVariableOp?
CNN_FCNN_Model/SMILES_Emb/CastCastsmiles_input*

DstT0*

SrcT0	*'
_output_shapes
:?????????H2 
CNN_FCNN_Model/SMILES_Emb/Cast?
*CNN_FCNN_Model/SMILES_Emb/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*CNN_FCNN_Model/SMILES_Emb/one_hot/on_value?
+CNN_FCNN_Model/SMILES_Emb/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+CNN_FCNN_Model/SMILES_Emb/one_hot/off_value?
'CNN_FCNN_Model/SMILES_Emb/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2)
'CNN_FCNN_Model/SMILES_Emb/one_hot/depth?
!CNN_FCNN_Model/SMILES_Emb/one_hotOneHot"CNN_FCNN_Model/SMILES_Emb/Cast:y:00CNN_FCNN_Model/SMILES_Emb/one_hot/depth:output:03CNN_FCNN_Model/SMILES_Emb/one_hot/on_value:output:04CNN_FCNN_Model/SMILES_Emb/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????H2#
!CNN_FCNN_Model/SMILES_Emb/one_hot?
*CNN_FCNN_Model/SMILES_Emb/GatherV2/indicesConst*
_output_shapes
:*
dtype0*}
valuetBr"h                        	   
                                                   2,
*CNN_FCNN_Model/SMILES_Emb/GatherV2/indices?
'CNN_FCNN_Model/SMILES_Emb/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'CNN_FCNN_Model/SMILES_Emb/GatherV2/axis?
"CNN_FCNN_Model/SMILES_Emb/GatherV2GatherV2*CNN_FCNN_Model/SMILES_Emb/one_hot:output:03CNN_FCNN_Model/SMILES_Emb/GatherV2/indices:output:00CNN_FCNN_Model/SMILES_Emb/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????H2$
"CNN_FCNN_Model/SMILES_Emb/GatherV2?
CNN_FCNN_Model/Prot_Emb/CastCastprotein_input*

DstT0*

SrcT0	*(
_output_shapes
:??????????
2
CNN_FCNN_Model/Prot_Emb/Cast?
(CNN_FCNN_Model/Prot_Emb/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(CNN_FCNN_Model/Prot_Emb/one_hot/on_value?
)CNN_FCNN_Model/Prot_Emb/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)CNN_FCNN_Model/Prot_Emb/one_hot/off_value?
%CNN_FCNN_Model/Prot_Emb/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2'
%CNN_FCNN_Model/Prot_Emb/one_hot/depth?
CNN_FCNN_Model/Prot_Emb/one_hotOneHot CNN_FCNN_Model/Prot_Emb/Cast:y:0.CNN_FCNN_Model/Prot_Emb/one_hot/depth:output:01CNN_FCNN_Model/Prot_Emb/one_hot/on_value:output:02CNN_FCNN_Model/Prot_Emb/one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:??????????
2!
CNN_FCNN_Model/Prot_Emb/one_hot?
(CNN_FCNN_Model/Prot_Emb/GatherV2/indicesConst*
_output_shapes
:*
dtype0*e
value\BZ"P                        	   
                                 2*
(CNN_FCNN_Model/Prot_Emb/GatherV2/indices?
%CNN_FCNN_Model/Prot_Emb/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%CNN_FCNN_Model/Prot_Emb/GatherV2/axis?
 CNN_FCNN_Model/Prot_Emb/GatherV2GatherV2(CNN_FCNN_Model/Prot_Emb/one_hot:output:01CNN_FCNN_Model/Prot_Emb/GatherV2/indices:output:0.CNN_FCNN_Model/Prot_Emb/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:??????????
2"
 CNN_FCNN_Model/Prot_Emb/GatherV2?
1CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims/dim?
-CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims
ExpandDims+CNN_FCNN_Model/SMILES_Emb/GatherV2:output:0:CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H2/
-CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims?
>CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGcnn_fcnn_model_smiles_cnn_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02@
>CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?
3CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/dim?
/CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1
ExpandDimsFCNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp:value:0<CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@21
/CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1?
"CNN_FCNN_Model/SMILES_CNN_0/conv1dConv2D6CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims:output:08CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2$
"CNN_FCNN_Model/SMILES_CNN_0/conv1d?
*CNN_FCNN_Model/SMILES_CNN_0/conv1d/SqueezeSqueeze+CNN_FCNN_Model/SMILES_CNN_0/conv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2,
*CNN_FCNN_Model/SMILES_CNN_0/conv1d/Squeeze?
2CNN_FCNN_Model/SMILES_CNN_0/BiasAdd/ReadVariableOpReadVariableOp;cnn_fcnn_model_smiles_cnn_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2CNN_FCNN_Model/SMILES_CNN_0/BiasAdd/ReadVariableOp?
#CNN_FCNN_Model/SMILES_CNN_0/BiasAddBiasAdd3CNN_FCNN_Model/SMILES_CNN_0/conv1d/Squeeze:output:0:CNN_FCNN_Model/SMILES_CNN_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2%
#CNN_FCNN_Model/SMILES_CNN_0/BiasAdd?
 CNN_FCNN_Model/SMILES_CNN_0/ReluRelu,CNN_FCNN_Model/SMILES_CNN_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2"
 CNN_FCNN_Model/SMILES_CNN_0/Relu?
/CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims/dim?
+CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims
ExpandDims)CNN_FCNN_Model/Prot_Emb/GatherV2:output:08CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2-
+CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims?
<CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEcnn_fcnn_model_prot_cnn_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?
1CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/dim?
-CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1
ExpandDimsDCNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1?
 CNN_FCNN_Model/Prot_CNN_0/conv1dConv2D4CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims:output:06CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2"
 CNN_FCNN_Model/Prot_CNN_0/conv1d?
(CNN_FCNN_Model/Prot_CNN_0/conv1d/SqueezeSqueeze)CNN_FCNN_Model/Prot_CNN_0/conv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2*
(CNN_FCNN_Model/Prot_CNN_0/conv1d/Squeeze?
0CNN_FCNN_Model/Prot_CNN_0/BiasAdd/ReadVariableOpReadVariableOp9cnn_fcnn_model_prot_cnn_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0CNN_FCNN_Model/Prot_CNN_0/BiasAdd/ReadVariableOp?
!CNN_FCNN_Model/Prot_CNN_0/BiasAddBiasAdd1CNN_FCNN_Model/Prot_CNN_0/conv1d/Squeeze:output:08CNN_FCNN_Model/Prot_CNN_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2#
!CNN_FCNN_Model/Prot_CNN_0/BiasAdd?
CNN_FCNN_Model/Prot_CNN_0/ReluRelu*CNN_FCNN_Model/Prot_CNN_0/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2 
CNN_FCNN_Model/Prot_CNN_0/Relu?
1CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims/dim?
-CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims
ExpandDims.CNN_FCNN_Model/SMILES_CNN_0/Relu:activations:0:CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2/
-CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims?
>CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGcnn_fcnn_model_smiles_cnn_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02@
>CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?
3CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/dim?
/CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1
ExpandDimsFCNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp:value:0<CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@21
/CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1?
"CNN_FCNN_Model/SMILES_CNN_1/conv1dConv2D6CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims:output:08CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2$
"CNN_FCNN_Model/SMILES_CNN_1/conv1d?
*CNN_FCNN_Model/SMILES_CNN_1/conv1d/SqueezeSqueeze+CNN_FCNN_Model/SMILES_CNN_1/conv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2,
*CNN_FCNN_Model/SMILES_CNN_1/conv1d/Squeeze?
2CNN_FCNN_Model/SMILES_CNN_1/BiasAdd/ReadVariableOpReadVariableOp;cnn_fcnn_model_smiles_cnn_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2CNN_FCNN_Model/SMILES_CNN_1/BiasAdd/ReadVariableOp?
#CNN_FCNN_Model/SMILES_CNN_1/BiasAddBiasAdd3CNN_FCNN_Model/SMILES_CNN_1/conv1d/Squeeze:output:0:CNN_FCNN_Model/SMILES_CNN_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2%
#CNN_FCNN_Model/SMILES_CNN_1/BiasAdd?
 CNN_FCNN_Model/SMILES_CNN_1/ReluRelu,CNN_FCNN_Model/SMILES_CNN_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2"
 CNN_FCNN_Model/SMILES_CNN_1/Relu?
/CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims/dim?
+CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims
ExpandDims,CNN_FCNN_Model/Prot_CNN_0/Relu:activations:08CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2-
+CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims?
<CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEcnn_fcnn_model_prot_cnn_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02>
<CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?
1CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/dim?
-CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1
ExpandDimsDCNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2/
-CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1?
 CNN_FCNN_Model/Prot_CNN_1/conv1dConv2D4CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims:output:06CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2"
 CNN_FCNN_Model/Prot_CNN_1/conv1d?
(CNN_FCNN_Model/Prot_CNN_1/conv1d/SqueezeSqueeze)CNN_FCNN_Model/Prot_CNN_1/conv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2*
(CNN_FCNN_Model/Prot_CNN_1/conv1d/Squeeze?
0CNN_FCNN_Model/Prot_CNN_1/BiasAdd/ReadVariableOpReadVariableOp9cnn_fcnn_model_prot_cnn_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0CNN_FCNN_Model/Prot_CNN_1/BiasAdd/ReadVariableOp?
!CNN_FCNN_Model/Prot_CNN_1/BiasAddBiasAdd1CNN_FCNN_Model/Prot_CNN_1/conv1d/Squeeze:output:08CNN_FCNN_Model/Prot_CNN_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2#
!CNN_FCNN_Model/Prot_CNN_1/BiasAdd?
CNN_FCNN_Model/Prot_CNN_1/ReluRelu*CNN_FCNN_Model/Prot_CNN_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2 
CNN_FCNN_Model/Prot_CNN_1/Relu?
1CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims/dim?
-CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims
ExpandDims.CNN_FCNN_Model/SMILES_CNN_1/Relu:activations:0:CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2/
-CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims?
>CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGcnn_fcnn_model_smiles_cnn_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02@
>CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?
3CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/dim?
/CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1
ExpandDimsFCNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp:value:0<CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?21
/CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1?
"CNN_FCNN_Model/SMILES_CNN_2/conv1dConv2D6CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims:output:08CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????H?*
paddingSAME*
strides
2$
"CNN_FCNN_Model/SMILES_CNN_2/conv1d?
*CNN_FCNN_Model/SMILES_CNN_2/conv1d/SqueezeSqueeze+CNN_FCNN_Model/SMILES_CNN_2/conv1d:output:0*
T0*,
_output_shapes
:?????????H?*
squeeze_dims

?????????2,
*CNN_FCNN_Model/SMILES_CNN_2/conv1d/Squeeze?
2CNN_FCNN_Model/SMILES_CNN_2/BiasAdd/ReadVariableOpReadVariableOp;cnn_fcnn_model_smiles_cnn_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2CNN_FCNN_Model/SMILES_CNN_2/BiasAdd/ReadVariableOp?
#CNN_FCNN_Model/SMILES_CNN_2/BiasAddBiasAdd3CNN_FCNN_Model/SMILES_CNN_2/conv1d/Squeeze:output:0:CNN_FCNN_Model/SMILES_CNN_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????H?2%
#CNN_FCNN_Model/SMILES_CNN_2/BiasAdd?
 CNN_FCNN_Model/SMILES_CNN_2/ReluRelu,CNN_FCNN_Model/SMILES_CNN_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????H?2"
 CNN_FCNN_Model/SMILES_CNN_2/Relu?
/CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims/dim?
+CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims
ExpandDims,CNN_FCNN_Model/Prot_CNN_1/Relu:activations:08CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2-
+CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims?
<CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEcnn_fcnn_model_prot_cnn_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02>
<CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?
1CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/dim?
-CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1
ExpandDimsDCNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2/
-CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1?
 CNN_FCNN_Model/Prot_CNN_2/conv1dConv2D4CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims:output:06CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:??????????
?*
paddingSAME*
strides
2"
 CNN_FCNN_Model/Prot_CNN_2/conv1d?
(CNN_FCNN_Model/Prot_CNN_2/conv1d/SqueezeSqueeze)CNN_FCNN_Model/Prot_CNN_2/conv1d:output:0*
T0*-
_output_shapes
:??????????
?*
squeeze_dims

?????????2*
(CNN_FCNN_Model/Prot_CNN_2/conv1d/Squeeze?
0CNN_FCNN_Model/Prot_CNN_2/BiasAdd/ReadVariableOpReadVariableOp9cnn_fcnn_model_prot_cnn_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0CNN_FCNN_Model/Prot_CNN_2/BiasAdd/ReadVariableOp?
!CNN_FCNN_Model/Prot_CNN_2/BiasAddBiasAdd1CNN_FCNN_Model/Prot_CNN_2/conv1d/Squeeze:output:08CNN_FCNN_Model/Prot_CNN_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????
?2#
!CNN_FCNN_Model/Prot_CNN_2/BiasAdd?
CNN_FCNN_Model/Prot_CNN_2/ReluRelu*CNN_FCNN_Model/Prot_CNN_2/BiasAdd:output:0*
T0*-
_output_shapes
:??????????
?2 
CNN_FCNN_Model/Prot_CNN_2/Relu?
4CNN_FCNN_Model/Prot_Global_Max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4CNN_FCNN_Model/Prot_Global_Max/Max/reduction_indices?
"CNN_FCNN_Model/Prot_Global_Max/MaxMax,CNN_FCNN_Model/Prot_CNN_2/Relu:activations:0=CNN_FCNN_Model/Prot_Global_Max/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2$
"CNN_FCNN_Model/Prot_Global_Max/Max?
6CNN_FCNN_Model/SMILES_Global_Max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :28
6CNN_FCNN_Model/SMILES_Global_Max/Max/reduction_indices?
$CNN_FCNN_Model/SMILES_Global_Max/MaxMax.CNN_FCNN_Model/SMILES_CNN_2/Relu:activations:0?CNN_FCNN_Model/SMILES_Global_Max/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2&
$CNN_FCNN_Model/SMILES_Global_Max/Max?
&CNN_FCNN_Model/Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&CNN_FCNN_Model/Concatenate/concat/axis?
!CNN_FCNN_Model/Concatenate/concatConcatV2+CNN_FCNN_Model/Prot_Global_Max/Max:output:0-CNN_FCNN_Model/SMILES_Global_Max/Max:output:0/CNN_FCNN_Model/Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2#
!CNN_FCNN_Model/Concatenate/concat?
,CNN_FCNN_Model/Dense_0/MatMul/ReadVariableOpReadVariableOp5cnn_fcnn_model_dense_0_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,CNN_FCNN_Model/Dense_0/MatMul/ReadVariableOp?
CNN_FCNN_Model/Dense_0/MatMulMatMul*CNN_FCNN_Model/Concatenate/concat:output:04CNN_FCNN_Model/Dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_FCNN_Model/Dense_0/MatMul?
-CNN_FCNN_Model/Dense_0/BiasAdd/ReadVariableOpReadVariableOp6cnn_fcnn_model_dense_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-CNN_FCNN_Model/Dense_0/BiasAdd/ReadVariableOp?
CNN_FCNN_Model/Dense_0/BiasAddBiasAdd'CNN_FCNN_Model/Dense_0/MatMul:product:05CNN_FCNN_Model/Dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
CNN_FCNN_Model/Dense_0/BiasAdd?
CNN_FCNN_Model/Dense_0/ReluRelu'CNN_FCNN_Model/Dense_0/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
CNN_FCNN_Model/Dense_0/Relu?
!CNN_FCNN_Model/Dropout_0/IdentityIdentity)CNN_FCNN_Model/Dense_0/Relu:activations:0*
T0*(
_output_shapes
:??????????2#
!CNN_FCNN_Model/Dropout_0/Identity?
,CNN_FCNN_Model/Dense_1/MatMul/ReadVariableOpReadVariableOp5cnn_fcnn_model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,CNN_FCNN_Model/Dense_1/MatMul/ReadVariableOp?
CNN_FCNN_Model/Dense_1/MatMulMatMul*CNN_FCNN_Model/Dropout_0/Identity:output:04CNN_FCNN_Model/Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_FCNN_Model/Dense_1/MatMul?
-CNN_FCNN_Model/Dense_1/BiasAdd/ReadVariableOpReadVariableOp6cnn_fcnn_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-CNN_FCNN_Model/Dense_1/BiasAdd/ReadVariableOp?
CNN_FCNN_Model/Dense_1/BiasAddBiasAdd'CNN_FCNN_Model/Dense_1/MatMul:product:05CNN_FCNN_Model/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
CNN_FCNN_Model/Dense_1/BiasAdd?
CNN_FCNN_Model/Dense_1/ReluRelu'CNN_FCNN_Model/Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
CNN_FCNN_Model/Dense_1/Relu?
!CNN_FCNN_Model/Dropout_1/IdentityIdentity)CNN_FCNN_Model/Dense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2#
!CNN_FCNN_Model/Dropout_1/Identity?
,CNN_FCNN_Model/Dense_2/MatMul/ReadVariableOpReadVariableOp5cnn_fcnn_model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,CNN_FCNN_Model/Dense_2/MatMul/ReadVariableOp?
CNN_FCNN_Model/Dense_2/MatMulMatMul*CNN_FCNN_Model/Dropout_1/Identity:output:04CNN_FCNN_Model/Dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_FCNN_Model/Dense_2/MatMul?
-CNN_FCNN_Model/Dense_2/BiasAdd/ReadVariableOpReadVariableOp6cnn_fcnn_model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-CNN_FCNN_Model/Dense_2/BiasAdd/ReadVariableOp?
CNN_FCNN_Model/Dense_2/BiasAddBiasAdd'CNN_FCNN_Model/Dense_2/MatMul:product:05CNN_FCNN_Model/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
CNN_FCNN_Model/Dense_2/BiasAdd?
CNN_FCNN_Model/Dense_2/ReluRelu'CNN_FCNN_Model/Dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
CNN_FCNN_Model/Dense_2/Relu?
*CNN_FCNN_Model/dense/MatMul/ReadVariableOpReadVariableOp3cnn_fcnn_model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*CNN_FCNN_Model/dense/MatMul/ReadVariableOp?
CNN_FCNN_Model/dense/MatMulMatMul)CNN_FCNN_Model/Dense_2/Relu:activations:02CNN_FCNN_Model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_FCNN_Model/dense/MatMul?
+CNN_FCNN_Model/dense/BiasAdd/ReadVariableOpReadVariableOp4cnn_fcnn_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+CNN_FCNN_Model/dense/BiasAdd/ReadVariableOp?
CNN_FCNN_Model/dense/BiasAddBiasAdd%CNN_FCNN_Model/dense/MatMul:product:03CNN_FCNN_Model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_FCNN_Model/dense/BiasAdd?	
IdentityIdentity%CNN_FCNN_Model/dense/BiasAdd:output:0.^CNN_FCNN_Model/Dense_0/BiasAdd/ReadVariableOp-^CNN_FCNN_Model/Dense_0/MatMul/ReadVariableOp.^CNN_FCNN_Model/Dense_1/BiasAdd/ReadVariableOp-^CNN_FCNN_Model/Dense_1/MatMul/ReadVariableOp.^CNN_FCNN_Model/Dense_2/BiasAdd/ReadVariableOp-^CNN_FCNN_Model/Dense_2/MatMul/ReadVariableOp1^CNN_FCNN_Model/Prot_CNN_0/BiasAdd/ReadVariableOp=^CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp1^CNN_FCNN_Model/Prot_CNN_1/BiasAdd/ReadVariableOp=^CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp1^CNN_FCNN_Model/Prot_CNN_2/BiasAdd/ReadVariableOp=^CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp3^CNN_FCNN_Model/SMILES_CNN_0/BiasAdd/ReadVariableOp?^CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp3^CNN_FCNN_Model/SMILES_CNN_1/BiasAdd/ReadVariableOp?^CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp3^CNN_FCNN_Model/SMILES_CNN_2/BiasAdd/ReadVariableOp?^CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp,^CNN_FCNN_Model/dense/BiasAdd/ReadVariableOp+^CNN_FCNN_Model/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::2^
-CNN_FCNN_Model/Dense_0/BiasAdd/ReadVariableOp-CNN_FCNN_Model/Dense_0/BiasAdd/ReadVariableOp2\
,CNN_FCNN_Model/Dense_0/MatMul/ReadVariableOp,CNN_FCNN_Model/Dense_0/MatMul/ReadVariableOp2^
-CNN_FCNN_Model/Dense_1/BiasAdd/ReadVariableOp-CNN_FCNN_Model/Dense_1/BiasAdd/ReadVariableOp2\
,CNN_FCNN_Model/Dense_1/MatMul/ReadVariableOp,CNN_FCNN_Model/Dense_1/MatMul/ReadVariableOp2^
-CNN_FCNN_Model/Dense_2/BiasAdd/ReadVariableOp-CNN_FCNN_Model/Dense_2/BiasAdd/ReadVariableOp2\
,CNN_FCNN_Model/Dense_2/MatMul/ReadVariableOp,CNN_FCNN_Model/Dense_2/MatMul/ReadVariableOp2d
0CNN_FCNN_Model/Prot_CNN_0/BiasAdd/ReadVariableOp0CNN_FCNN_Model/Prot_CNN_0/BiasAdd/ReadVariableOp2|
<CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp<CNN_FCNN_Model/Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp2d
0CNN_FCNN_Model/Prot_CNN_1/BiasAdd/ReadVariableOp0CNN_FCNN_Model/Prot_CNN_1/BiasAdd/ReadVariableOp2|
<CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp<CNN_FCNN_Model/Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp2d
0CNN_FCNN_Model/Prot_CNN_2/BiasAdd/ReadVariableOp0CNN_FCNN_Model/Prot_CNN_2/BiasAdd/ReadVariableOp2|
<CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp<CNN_FCNN_Model/Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp2h
2CNN_FCNN_Model/SMILES_CNN_0/BiasAdd/ReadVariableOp2CNN_FCNN_Model/SMILES_CNN_0/BiasAdd/ReadVariableOp2?
>CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp>CNN_FCNN_Model/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp2h
2CNN_FCNN_Model/SMILES_CNN_1/BiasAdd/ReadVariableOp2CNN_FCNN_Model/SMILES_CNN_1/BiasAdd/ReadVariableOp2?
>CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp>CNN_FCNN_Model/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp2h
2CNN_FCNN_Model/SMILES_CNN_2/BiasAdd/ReadVariableOp2CNN_FCNN_Model/SMILES_CNN_2/BiasAdd/ReadVariableOp2?
>CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp>CNN_FCNN_Model/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp2Z
+CNN_FCNN_Model/dense/BiasAdd/ReadVariableOp+CNN_FCNN_Model/dense/BiasAdd/ReadVariableOp2X
*CNN_FCNN_Model/dense/MatMul/ReadVariableOp*CNN_FCNN_Model/dense/MatMul/ReadVariableOp:W S
(
_output_shapes
:??????????

'
_user_specified_nameProtein_Input:UQ
'
_output_shapes
:?????????H
&
_user_specified_nameSMILES_Input
ϲ
?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_3471
inputs_0	
inputs_1	<
8smiles_cnn_0_conv1d_expanddims_1_readvariableop_resource0
,smiles_cnn_0_biasadd_readvariableop_resource:
6prot_cnn_0_conv1d_expanddims_1_readvariableop_resource.
*prot_cnn_0_biasadd_readvariableop_resource<
8smiles_cnn_1_conv1d_expanddims_1_readvariableop_resource0
,smiles_cnn_1_biasadd_readvariableop_resource:
6prot_cnn_1_conv1d_expanddims_1_readvariableop_resource.
*prot_cnn_1_biasadd_readvariableop_resource<
8smiles_cnn_2_conv1d_expanddims_1_readvariableop_resource0
,smiles_cnn_2_biasadd_readvariableop_resource:
6prot_cnn_2_conv1d_expanddims_1_readvariableop_resource.
*prot_cnn_2_biasadd_readvariableop_resource*
&dense_0_matmul_readvariableop_resource+
'dense_0_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??Dense_0/BiasAdd/ReadVariableOp?Dense_0/MatMul/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?!Prot_CNN_0/BiasAdd/ReadVariableOp?-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?!Prot_CNN_1/BiasAdd/ReadVariableOp?-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?!Prot_CNN_2/BiasAdd/ReadVariableOp?-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?#SMILES_CNN_0/BiasAdd/ReadVariableOp?/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?#SMILES_CNN_1/BiasAdd/ReadVariableOp?/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?#SMILES_CNN_2/BiasAdd/ReadVariableOp?/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOpu
SMILES_Emb/CastCastinputs_1*

DstT0*

SrcT0	*'
_output_shapes
:?????????H2
SMILES_Emb/Cast
SMILES_Emb/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
SMILES_Emb/one_hot/on_value?
SMILES_Emb/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SMILES_Emb/one_hot/off_valuev
SMILES_Emb/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
SMILES_Emb/one_hot/depth?
SMILES_Emb/one_hotOneHotSMILES_Emb/Cast:y:0!SMILES_Emb/one_hot/depth:output:0$SMILES_Emb/one_hot/on_value:output:0%SMILES_Emb/one_hot/off_value:output:0*
T0*
TI0*+
_output_shapes
:?????????H2
SMILES_Emb/one_hot?
SMILES_Emb/GatherV2/indicesConst*
_output_shapes
:*
dtype0*}
valuetBr"h                        	   
                                                   2
SMILES_Emb/GatherV2/indicesv
SMILES_Emb/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
SMILES_Emb/GatherV2/axis?
SMILES_Emb/GatherV2GatherV2SMILES_Emb/one_hot:output:0$SMILES_Emb/GatherV2/indices:output:0!SMILES_Emb/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????H2
SMILES_Emb/GatherV2r
Prot_Emb/CastCastinputs_0*

DstT0*

SrcT0	*(
_output_shapes
:??????????
2
Prot_Emb/Cast{
Prot_Emb/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
Prot_Emb/one_hot/on_value}
Prot_Emb/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Prot_Emb/one_hot/off_valuer
Prot_Emb/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :2
Prot_Emb/one_hot/depth?
Prot_Emb/one_hotOneHotProt_Emb/Cast:y:0Prot_Emb/one_hot/depth:output:0"Prot_Emb/one_hot/on_value:output:0#Prot_Emb/one_hot/off_value:output:0*
T0*
TI0*,
_output_shapes
:??????????
2
Prot_Emb/one_hot?
Prot_Emb/GatherV2/indicesConst*
_output_shapes
:*
dtype0*e
value\BZ"P                        	   
                                 2
Prot_Emb/GatherV2/indicesr
Prot_Emb/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Prot_Emb/GatherV2/axis?
Prot_Emb/GatherV2GatherV2Prot_Emb/one_hot:output:0"Prot_Emb/GatherV2/indices:output:0Prot_Emb/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:??????????
2
Prot_Emb/GatherV2?
"SMILES_CNN_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"SMILES_CNN_0/conv1d/ExpandDims/dim?
SMILES_CNN_0/conv1d/ExpandDims
ExpandDimsSMILES_Emb/GatherV2:output:0+SMILES_CNN_0/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H2 
SMILES_CNN_0/conv1d/ExpandDims?
/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8smiles_cnn_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype021
/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?
$SMILES_CNN_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$SMILES_CNN_0/conv1d/ExpandDims_1/dim?
 SMILES_CNN_0/conv1d/ExpandDims_1
ExpandDims7SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp:value:0-SMILES_CNN_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2"
 SMILES_CNN_0/conv1d/ExpandDims_1?
SMILES_CNN_0/conv1dConv2D'SMILES_CNN_0/conv1d/ExpandDims:output:0)SMILES_CNN_0/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2
SMILES_CNN_0/conv1d?
SMILES_CNN_0/conv1d/SqueezeSqueezeSMILES_CNN_0/conv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2
SMILES_CNN_0/conv1d/Squeeze?
#SMILES_CNN_0/BiasAdd/ReadVariableOpReadVariableOp,smiles_cnn_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#SMILES_CNN_0/BiasAdd/ReadVariableOp?
SMILES_CNN_0/BiasAddBiasAdd$SMILES_CNN_0/conv1d/Squeeze:output:0+SMILES_CNN_0/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2
SMILES_CNN_0/BiasAdd?
SMILES_CNN_0/ReluReluSMILES_CNN_0/BiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2
SMILES_CNN_0/Relu?
 Prot_CNN_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 Prot_CNN_0/conv1d/ExpandDims/dim?
Prot_CNN_0/conv1d/ExpandDims
ExpandDimsProt_Emb/GatherV2:output:0)Prot_CNN_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
Prot_CNN_0/conv1d/ExpandDims?
-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6prot_cnn_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02/
-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp?
"Prot_CNN_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Prot_CNN_0/conv1d/ExpandDims_1/dim?
Prot_CNN_0/conv1d/ExpandDims_1
ExpandDims5Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp:value:0+Prot_CNN_0/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2 
Prot_CNN_0/conv1d/ExpandDims_1?
Prot_CNN_0/conv1dConv2D%Prot_CNN_0/conv1d/ExpandDims:output:0'Prot_CNN_0/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2
Prot_CNN_0/conv1d?
Prot_CNN_0/conv1d/SqueezeSqueezeProt_CNN_0/conv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2
Prot_CNN_0/conv1d/Squeeze?
!Prot_CNN_0/BiasAdd/ReadVariableOpReadVariableOp*prot_cnn_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Prot_CNN_0/BiasAdd/ReadVariableOp?
Prot_CNN_0/BiasAddBiasAdd"Prot_CNN_0/conv1d/Squeeze:output:0)Prot_CNN_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2
Prot_CNN_0/BiasAdd~
Prot_CNN_0/ReluReluProt_CNN_0/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2
Prot_CNN_0/Relu?
"SMILES_CNN_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"SMILES_CNN_1/conv1d/ExpandDims/dim?
SMILES_CNN_1/conv1d/ExpandDims
ExpandDimsSMILES_CNN_0/Relu:activations:0+SMILES_CNN_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2 
SMILES_CNN_1/conv1d/ExpandDims?
/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8smiles_cnn_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype021
/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?
$SMILES_CNN_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$SMILES_CNN_1/conv1d/ExpandDims_1/dim?
 SMILES_CNN_1/conv1d/ExpandDims_1
ExpandDims7SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp:value:0-SMILES_CNN_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2"
 SMILES_CNN_1/conv1d/ExpandDims_1?
SMILES_CNN_1/conv1dConv2D'SMILES_CNN_1/conv1d/ExpandDims:output:0)SMILES_CNN_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????H@*
paddingSAME*
strides
2
SMILES_CNN_1/conv1d?
SMILES_CNN_1/conv1d/SqueezeSqueezeSMILES_CNN_1/conv1d:output:0*
T0*+
_output_shapes
:?????????H@*
squeeze_dims

?????????2
SMILES_CNN_1/conv1d/Squeeze?
#SMILES_CNN_1/BiasAdd/ReadVariableOpReadVariableOp,smiles_cnn_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#SMILES_CNN_1/BiasAdd/ReadVariableOp?
SMILES_CNN_1/BiasAddBiasAdd$SMILES_CNN_1/conv1d/Squeeze:output:0+SMILES_CNN_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????H@2
SMILES_CNN_1/BiasAdd?
SMILES_CNN_1/ReluReluSMILES_CNN_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????H@2
SMILES_CNN_1/Relu?
 Prot_CNN_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 Prot_CNN_1/conv1d/ExpandDims/dim?
Prot_CNN_1/conv1d/ExpandDims
ExpandDimsProt_CNN_0/Relu:activations:0)Prot_CNN_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2
Prot_CNN_1/conv1d/ExpandDims?
-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6prot_cnn_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp?
"Prot_CNN_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Prot_CNN_1/conv1d/ExpandDims_1/dim?
Prot_CNN_1/conv1d/ExpandDims_1
ExpandDims5Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp:value:0+Prot_CNN_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2 
Prot_CNN_1/conv1d/ExpandDims_1?
Prot_CNN_1/conv1dConv2D%Prot_CNN_1/conv1d/ExpandDims:output:0'Prot_CNN_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
@*
paddingSAME*
strides
2
Prot_CNN_1/conv1d?
Prot_CNN_1/conv1d/SqueezeSqueezeProt_CNN_1/conv1d:output:0*
T0*,
_output_shapes
:??????????
@*
squeeze_dims

?????????2
Prot_CNN_1/conv1d/Squeeze?
!Prot_CNN_1/BiasAdd/ReadVariableOpReadVariableOp*prot_cnn_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Prot_CNN_1/BiasAdd/ReadVariableOp?
Prot_CNN_1/BiasAddBiasAdd"Prot_CNN_1/conv1d/Squeeze:output:0)Prot_CNN_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
@2
Prot_CNN_1/BiasAdd~
Prot_CNN_1/ReluReluProt_CNN_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????
@2
Prot_CNN_1/Relu?
"SMILES_CNN_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"SMILES_CNN_2/conv1d/ExpandDims/dim?
SMILES_CNN_2/conv1d/ExpandDims
ExpandDimsSMILES_CNN_1/Relu:activations:0+SMILES_CNN_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????H@2 
SMILES_CNN_2/conv1d/ExpandDims?
/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8smiles_cnn_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype021
/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?
$SMILES_CNN_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$SMILES_CNN_2/conv1d/ExpandDims_1/dim?
 SMILES_CNN_2/conv1d/ExpandDims_1
ExpandDims7SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp:value:0-SMILES_CNN_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2"
 SMILES_CNN_2/conv1d/ExpandDims_1?
SMILES_CNN_2/conv1dConv2D'SMILES_CNN_2/conv1d/ExpandDims:output:0)SMILES_CNN_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????H?*
paddingSAME*
strides
2
SMILES_CNN_2/conv1d?
SMILES_CNN_2/conv1d/SqueezeSqueezeSMILES_CNN_2/conv1d:output:0*
T0*,
_output_shapes
:?????????H?*
squeeze_dims

?????????2
SMILES_CNN_2/conv1d/Squeeze?
#SMILES_CNN_2/BiasAdd/ReadVariableOpReadVariableOp,smiles_cnn_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#SMILES_CNN_2/BiasAdd/ReadVariableOp?
SMILES_CNN_2/BiasAddBiasAdd$SMILES_CNN_2/conv1d/Squeeze:output:0+SMILES_CNN_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????H?2
SMILES_CNN_2/BiasAdd?
SMILES_CNN_2/ReluReluSMILES_CNN_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????H?2
SMILES_CNN_2/Relu?
 Prot_CNN_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 Prot_CNN_2/conv1d/ExpandDims/dim?
Prot_CNN_2/conv1d/ExpandDims
ExpandDimsProt_CNN_1/Relu:activations:0)Prot_CNN_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
@2
Prot_CNN_2/conv1d/ExpandDims?
-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6prot_cnn_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02/
-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp?
"Prot_CNN_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Prot_CNN_2/conv1d/ExpandDims_1/dim?
Prot_CNN_2/conv1d/ExpandDims_1
ExpandDims5Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp:value:0+Prot_CNN_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2 
Prot_CNN_2/conv1d/ExpandDims_1?
Prot_CNN_2/conv1dConv2D%Prot_CNN_2/conv1d/ExpandDims:output:0'Prot_CNN_2/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:??????????
?*
paddingSAME*
strides
2
Prot_CNN_2/conv1d?
Prot_CNN_2/conv1d/SqueezeSqueezeProt_CNN_2/conv1d:output:0*
T0*-
_output_shapes
:??????????
?*
squeeze_dims

?????????2
Prot_CNN_2/conv1d/Squeeze?
!Prot_CNN_2/BiasAdd/ReadVariableOpReadVariableOp*prot_cnn_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!Prot_CNN_2/BiasAdd/ReadVariableOp?
Prot_CNN_2/BiasAddBiasAdd"Prot_CNN_2/conv1d/Squeeze:output:0)Prot_CNN_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????
?2
Prot_CNN_2/BiasAdd
Prot_CNN_2/ReluReluProt_CNN_2/BiasAdd:output:0*
T0*-
_output_shapes
:??????????
?2
Prot_CNN_2/Relu?
%Prot_Global_Max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%Prot_Global_Max/Max/reduction_indices?
Prot_Global_Max/MaxMaxProt_CNN_2/Relu:activations:0.Prot_Global_Max/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Prot_Global_Max/Max?
'SMILES_Global_Max/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2)
'SMILES_Global_Max/Max/reduction_indices?
SMILES_Global_Max/MaxMaxSMILES_CNN_2/Relu:activations:00SMILES_Global_Max/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
SMILES_Global_Max/Maxt
Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concatenate/concat/axis?
Concatenate/concatConcatV2Prot_Global_Max/Max:output:0SMILES_Global_Max/Max:output:0 Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
Concatenate/concat?
Dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_0/MatMul/ReadVariableOp?
Dense_0/MatMulMatMulConcatenate/concat:output:0%Dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_0/MatMul?
Dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_0/BiasAdd/ReadVariableOp?
Dense_0/BiasAddBiasAddDense_0/MatMul:product:0&Dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_0/BiasAddq
Dense_0/ReluReluDense_0/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_0/Relu?
Dropout_0/IdentityIdentityDense_0/Relu:activations:0*
T0*(
_output_shapes
:??????????2
Dropout_0/Identity?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMulDropout_0/Identity:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_1/BiasAddq
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_1/Relu?
Dropout_1/IdentityIdentityDense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
Dropout_1/Identity?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMulDropout_1/Identity:output:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_2/BiasAddq
Dense_2/ReluReluDense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_2/Relu?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulDense_2/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^Dense_0/BiasAdd/ReadVariableOp^Dense_0/MatMul/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp"^Prot_CNN_0/BiasAdd/ReadVariableOp.^Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp"^Prot_CNN_1/BiasAdd/ReadVariableOp.^Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp"^Prot_CNN_2/BiasAdd/ReadVariableOp.^Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp$^SMILES_CNN_0/BiasAdd/ReadVariableOp0^SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp$^SMILES_CNN_1/BiasAdd/ReadVariableOp0^SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp$^SMILES_CNN_2/BiasAdd/ReadVariableOp0^SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::2@
Dense_0/BiasAdd/ReadVariableOpDense_0/BiasAdd/ReadVariableOp2>
Dense_0/MatMul/ReadVariableOpDense_0/MatMul/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp2F
!Prot_CNN_0/BiasAdd/ReadVariableOp!Prot_CNN_0/BiasAdd/ReadVariableOp2^
-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp-Prot_CNN_0/conv1d/ExpandDims_1/ReadVariableOp2F
!Prot_CNN_1/BiasAdd/ReadVariableOp!Prot_CNN_1/BiasAdd/ReadVariableOp2^
-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp-Prot_CNN_1/conv1d/ExpandDims_1/ReadVariableOp2F
!Prot_CNN_2/BiasAdd/ReadVariableOp!Prot_CNN_2/BiasAdd/ReadVariableOp2^
-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp-Prot_CNN_2/conv1d/ExpandDims_1/ReadVariableOp2J
#SMILES_CNN_0/BiasAdd/ReadVariableOp#SMILES_CNN_0/BiasAdd/ReadVariableOp2b
/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp/SMILES_CNN_0/conv1d/ExpandDims_1/ReadVariableOp2J
#SMILES_CNN_1/BiasAdd/ReadVariableOp#SMILES_CNN_1/BiasAdd/ReadVariableOp2b
/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp/SMILES_CNN_1/conv1d/ExpandDims_1/ReadVariableOp2J
#SMILES_CNN_2/BiasAdd/ReadVariableOp#SMILES_CNN_2/BiasAdd/ReadVariableOp2b
/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp/SMILES_CNN_2/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????H
"
_user_specified_name
inputs/1
?
?
-__inference_CNN_FCNN_Model_layer_call_fn_3045
protein_input	
smiles_input	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprotein_inputsmiles_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_30022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????

'
_user_specified_nameProtein_Input:UQ
'
_output_shapes
:?????????H
&
_user_specified_nameSMILES_Input
?
?
-__inference_CNN_FCNN_Model_layer_call_fn_3517
inputs_0	
inputs_1	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_30022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesy
w:??????????
:?????????H::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????H
"
_user_specified_name
inputs/1
?	
?
A__inference_Dense_2_layer_call_and_return_conditional_losses_2831

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
H
Protein_Input7
serving_default_Protein_Input:0	??????????

E
SMILES_Input5
serving_default_SMILES_Input:0	?????????H9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
ְ
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"??
_tf_keras_network??{"class_name": "Functional", "name": "CNN_FCNN_Model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_FCNN_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "Protein_Input"}, "name": "Protein_Input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "SMILES_Input"}, "name": "SMILES_Input", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "Prot_Emb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wIAAAAAAAAAAwAAAAYAAABTAAAAczoAAAB0AGoBdACgAnwAZAGhAnwBZAKNAn0CdABqA3wCZANk\nBIQAdARkBXwBgwJEAIMBZAZkB40DfQJ8AlMAKQhO2gV1aW50OCkB2gVkZXB0aGMBAAAAAAAAAAIA\nAAADAAAAUwAAAHMQAAAAZwB8AF0IfQF8AZECcQRTAKkAcgMAAAApAtoCLjDaAWlyAwAAAHIDAAAA\n+ipDOi9QaERUaGVzaXMvU3JjL0dyYWRSQU0vY25uX2Zjbm5fbW9kZWwucHn6CjxsaXN0Y29tcD4P\nAAAAcwIAAAAGAHpFZ2VuZXJhdGVfb25lX2hvdF9sYXllci48bG9jYWxzPi5vbmVfaG90X2VuY29u\nZGluZy48bG9jYWxzPi48bGlzdGNvbXA+6QEAAADpAgAAACkB2gRheGlzKQXaAnRm2gdvbmVfaG90\n2gRjYXN02gZnYXRoZXLaBXJhbmdlKQPaAXjaC251bV9jbGFzc2VzcgwAAAByAwAAAHIDAAAAcgYA\nAADaEW9uZV9ob3RfZW5jb25kaW5nDAAAAHMIAAAAAAEWAQQBHAE=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"num_classes": 21}}, "name": "Prot_Emb", "inbound_nodes": [[["Protein_Input", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "SMILES_Emb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wIAAAAAAAAAAwAAAAYAAABTAAAAczoAAAB0AGoBdACgAnwAZAGhAnwBZAKNAn0CdABqA3wCZANk\nBIQAdARkBXwBgwJEAIMBZAZkB40DfQJ8AlMAKQhO2gV1aW50OCkB2gVkZXB0aGMBAAAAAAAAAAIA\nAAADAAAAUwAAAHMQAAAAZwB8AF0IfQF8AZECcQRTAKkAcgMAAAApAtoCLjDaAWlyAwAAAHIDAAAA\n+ipDOi9QaERUaGVzaXMvU3JjL0dyYWRSQU0vY25uX2Zjbm5fbW9kZWwucHn6CjxsaXN0Y29tcD4P\nAAAAcwIAAAAGAHpFZ2VuZXJhdGVfb25lX2hvdF9sYXllci48bG9jYWxzPi5vbmVfaG90X2VuY29u\nZGluZy48bG9jYWxzPi48bGlzdGNvbXA+6QEAAADpAgAAACkB2gRheGlzKQXaAnRm2gdvbmVfaG90\n2gRjYXN02gZnYXRoZXLaBXJhbmdlKQPaAXjaC251bV9jbGFzc2VzcgwAAAByAwAAAHIDAAAAcgYA\nAADaEW9uZV9ob3RfZW5jb25kaW5nDAAAAHMIAAAAAAEWAQQBHAE=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"num_classes": 27}}, "name": "SMILES_Emb", "inbound_nodes": [[["SMILES_Input", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Prot_CNN_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prot_CNN_0", "inbound_nodes": [[["Prot_Emb", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "SMILES_CNN_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SMILES_CNN_0", "inbound_nodes": [[["SMILES_Emb", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Prot_CNN_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prot_CNN_1", "inbound_nodes": [[["Prot_CNN_0", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "SMILES_CNN_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SMILES_CNN_1", "inbound_nodes": [[["SMILES_CNN_0", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Prot_CNN_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prot_CNN_2", "inbound_nodes": [[["Prot_CNN_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "SMILES_CNN_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SMILES_CNN_2", "inbound_nodes": [[["SMILES_CNN_1", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Prot_Global_Max", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Prot_Global_Max", "inbound_nodes": [[["Prot_CNN_2", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "SMILES_Global_Max", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "SMILES_Global_Max", "inbound_nodes": [[["SMILES_CNN_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate", "inbound_nodes": [[["Prot_Global_Max", 0, 0, {}], ["SMILES_Global_Max", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_0", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_0", "inbound_nodes": [[["Concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout_0", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "Dropout_0", "inbound_nodes": [[["Dense_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["Dropout_0", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Dropout_1", "inbound_nodes": [[["Dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["Dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["Dense_2", 0, 0, {}]]]}], "input_layers": [["Protein_Input", 0, 0], ["SMILES_Input", 0, 0]], "output_layers": [["dense", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1400]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 72]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1400]}, {"class_name": "TensorShape", "items": [null, 72]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_FCNN_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "Protein_Input"}, "name": "Protein_Input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "SMILES_Input"}, "name": "SMILES_Input", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "Prot_Emb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wIAAAAAAAAAAwAAAAYAAABTAAAAczoAAAB0AGoBdACgAnwAZAGhAnwBZAKNAn0CdABqA3wCZANk\nBIQAdARkBXwBgwJEAIMBZAZkB40DfQJ8AlMAKQhO2gV1aW50OCkB2gVkZXB0aGMBAAAAAAAAAAIA\nAAADAAAAUwAAAHMQAAAAZwB8AF0IfQF8AZECcQRTAKkAcgMAAAApAtoCLjDaAWlyAwAAAHIDAAAA\n+ipDOi9QaERUaGVzaXMvU3JjL0dyYWRSQU0vY25uX2Zjbm5fbW9kZWwucHn6CjxsaXN0Y29tcD4P\nAAAAcwIAAAAGAHpFZ2VuZXJhdGVfb25lX2hvdF9sYXllci48bG9jYWxzPi5vbmVfaG90X2VuY29u\nZGluZy48bG9jYWxzPi48bGlzdGNvbXA+6QEAAADpAgAAACkB2gRheGlzKQXaAnRm2gdvbmVfaG90\n2gRjYXN02gZnYXRoZXLaBXJhbmdlKQPaAXjaC251bV9jbGFzc2VzcgwAAAByAwAAAHIDAAAAcgYA\nAADaEW9uZV9ob3RfZW5jb25kaW5nDAAAAHMIAAAAAAEWAQQBHAE=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"num_classes": 21}}, "name": "Prot_Emb", "inbound_nodes": [[["Protein_Input", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "SMILES_Emb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wIAAAAAAAAAAwAAAAYAAABTAAAAczoAAAB0AGoBdACgAnwAZAGhAnwBZAKNAn0CdABqA3wCZANk\nBIQAdARkBXwBgwJEAIMBZAZkB40DfQJ8AlMAKQhO2gV1aW50OCkB2gVkZXB0aGMBAAAAAAAAAAIA\nAAADAAAAUwAAAHMQAAAAZwB8AF0IfQF8AZECcQRTAKkAcgMAAAApAtoCLjDaAWlyAwAAAHIDAAAA\n+ipDOi9QaERUaGVzaXMvU3JjL0dyYWRSQU0vY25uX2Zjbm5fbW9kZWwucHn6CjxsaXN0Y29tcD4P\nAAAAcwIAAAAGAHpFZ2VuZXJhdGVfb25lX2hvdF9sYXllci48bG9jYWxzPi5vbmVfaG90X2VuY29u\nZGluZy48bG9jYWxzPi48bGlzdGNvbXA+6QEAAADpAgAAACkB2gRheGlzKQXaAnRm2gdvbmVfaG90\n2gRjYXN02gZnYXRoZXLaBXJhbmdlKQPaAXjaC251bV9jbGFzc2VzcgwAAAByAwAAAHIDAAAAcgYA\nAADaEW9uZV9ob3RfZW5jb25kaW5nDAAAAHMIAAAAAAEWAQQBHAE=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"num_classes": 27}}, "name": "SMILES_Emb", "inbound_nodes": [[["SMILES_Input", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Prot_CNN_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prot_CNN_0", "inbound_nodes": [[["Prot_Emb", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "SMILES_CNN_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SMILES_CNN_0", "inbound_nodes": [[["SMILES_Emb", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Prot_CNN_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prot_CNN_1", "inbound_nodes": [[["Prot_CNN_0", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "SMILES_CNN_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SMILES_CNN_1", "inbound_nodes": [[["SMILES_CNN_0", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Prot_CNN_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prot_CNN_2", "inbound_nodes": [[["Prot_CNN_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "SMILES_CNN_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "SMILES_CNN_2", "inbound_nodes": [[["SMILES_CNN_1", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Prot_Global_Max", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Prot_Global_Max", "inbound_nodes": [[["Prot_CNN_2", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "SMILES_Global_Max", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "SMILES_Global_Max", "inbound_nodes": [[["SMILES_CNN_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenate", "inbound_nodes": [[["Prot_Global_Max", 0, 0, {}], ["SMILES_Global_Max", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_0", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_0", "inbound_nodes": [[["Concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout_0", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "Dropout_0", "inbound_nodes": [[["Dense_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["Dropout_0", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Dropout_1", "inbound_nodes": [[["Dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["Dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["Dense_2", 0, 0, {}]]]}], "input_layers": [["Protein_Input", 0, 0], ["SMILES_Input", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "RootMeanSquaredError", "config": {"name": "root_mean_squared_error", "dtype": "float32"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "c_index", "dtype": "float32", "fn": "c_index"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "Protein_Input", "dtype": "int64", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "Protein_Input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "SMILES_Input", "dtype": "int64", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "int64", "sparse": false, "ragged": false, "name": "SMILES_Input"}}
?

regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Lambda", "name": "Prot_Emb", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "Prot_Emb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wIAAAAAAAAAAwAAAAYAAABTAAAAczoAAAB0AGoBdACgAnwAZAGhAnwBZAKNAn0CdABqA3wCZANk\nBIQAdARkBXwBgwJEAIMBZAZkB40DfQJ8AlMAKQhO2gV1aW50OCkB2gVkZXB0aGMBAAAAAAAAAAIA\nAAADAAAAUwAAAHMQAAAAZwB8AF0IfQF8AZECcQRTAKkAcgMAAAApAtoCLjDaAWlyAwAAAHIDAAAA\n+ipDOi9QaERUaGVzaXMvU3JjL0dyYWRSQU0vY25uX2Zjbm5fbW9kZWwucHn6CjxsaXN0Y29tcD4P\nAAAAcwIAAAAGAHpFZ2VuZXJhdGVfb25lX2hvdF9sYXllci48bG9jYWxzPi5vbmVfaG90X2VuY29u\nZGluZy48bG9jYWxzPi48bGlzdGNvbXA+6QEAAADpAgAAACkB2gRheGlzKQXaAnRm2gdvbmVfaG90\n2gRjYXN02gZnYXRoZXLaBXJhbmdlKQPaAXjaC251bV9jbGFzc2VzcgwAAAByAwAAAHIDAAAAcgYA\nAADaEW9uZV9ob3RfZW5jb25kaW5nDAAAAHMIAAAAAAEWAQQBHAE=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"num_classes": 21}}}
?

regularization_losses
	variables
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Lambda", "name": "SMILES_Emb", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "SMILES_Emb", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wIAAAAAAAAAAwAAAAYAAABTAAAAczoAAAB0AGoBdACgAnwAZAGhAnwBZAKNAn0CdABqA3wCZANk\nBIQAdARkBXwBgwJEAIMBZAZkB40DfQJ8AlMAKQhO2gV1aW50OCkB2gVkZXB0aGMBAAAAAAAAAAIA\nAAADAAAAUwAAAHMQAAAAZwB8AF0IfQF8AZECcQRTAKkAcgMAAAApAtoCLjDaAWlyAwAAAHIDAAAA\n+ipDOi9QaERUaGVzaXMvU3JjL0dyYWRSQU0vY25uX2Zjbm5fbW9kZWwucHn6CjxsaXN0Y29tcD4P\nAAAAcwIAAAAGAHpFZ2VuZXJhdGVfb25lX2hvdF9sYXllci48bG9jYWxzPi5vbmVfaG90X2VuY29u\nZGluZy48bG9jYWxzPi48bGlzdGNvbXA+6QEAAADpAgAAACkB2gRheGlzKQXaAnRm2gdvbmVfaG90\n2gRjYXN02gZnYXRoZXLaBXJhbmdlKQPaAXjaC251bV9jbGFzc2VzcgwAAAByAwAAAHIDAAAAcgYA\nAADaEW9uZV9ob3RfZW5jb25kaW5nDAAAAHMIAAAAAAEWAQQBHAE=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {"num_classes": 27}}}
?	

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "Prot_CNN_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Prot_CNN_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1400, 20]}}
?	

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "SMILES_CNN_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SMILES_CNN_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 26}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 26]}}
?	

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "Prot_CNN_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Prot_CNN_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1400, 64]}}
?	

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "SMILES_CNN_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SMILES_CNN_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 64]}}
?	

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "Prot_CNN_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Prot_CNN_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1400, 64]}}
?	

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "SMILES_CNN_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SMILES_CNN_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 64]}}
?
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GlobalMaxPooling1D", "name": "Prot_Global_Max", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Prot_Global_Max", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GlobalMaxPooling1D", "name": "SMILES_Global_Max", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SMILES_Global_Max", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "Concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128]}, {"class_name": "TensorShape", "items": [null, 128]}]}
?

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_0", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "Dropout_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dropout_0", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

\kernel
]bias
^regularization_losses
_	variables
`trainable_variables
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
?
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "Dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Rm?Sm?\m?]m?fm?gm?lm?mm?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Rv?Sv?\v?]v?fv?gv?lv?mv?"
	optimizer
 "
trackable_list_wrapper
?
"0
#1
(2
)3
.4
/5
46
57
:8
;9
@10
A11
R12
S13
\14
]15
f16
g17
l18
m19"
trackable_list_wrapper
?
"0
#1
(2
)3
.4
/5
46
57
:8
;9
@10
A11
R12
S13
\14
]15
f16
g17
l18
m19"
trackable_list_wrapper
?
rlayer_metrics

slayers
tlayer_regularization_losses
umetrics
regularization_losses
	variables
trainable_variables
vnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables

xlayers
ylayer_regularization_losses
zmetrics
regularization_losses
	variables
trainable_variables
{layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~layer_regularization_losses
metrics
regularization_losses
	variables
 trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@2Prot_CNN_0/kernel
:@2Prot_CNN_0/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
$regularization_losses
%	variables
&trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@2SMILES_CNN_0/kernel
:@2SMILES_CNN_0/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
*regularization_losses
+	variables
,trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@@2Prot_CNN_1/kernel
:@2Prot_CNN_1/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
0regularization_losses
1	variables
2trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@@2SMILES_CNN_1/kernel
:@2SMILES_CNN_1/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
6regularization_losses
7	variables
8trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&@?2Prot_CNN_2/kernel
:?2Prot_CNN_2/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
<regularization_losses
=	variables
>trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@?2SMILES_CNN_2/kernel
 :?2SMILES_CNN_2/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Bregularization_losses
C	variables
Dtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Fregularization_losses
G	variables
Htrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Jregularization_losses
K	variables
Ltrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Nregularization_losses
O	variables
Ptrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2Dense_0/kernel
:?2Dense_0/bias
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Tregularization_losses
U	variables
Vtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
Xregularization_losses
Y	variables
Ztrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2Dense_1/kernel
:?2Dense_1/bias
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
^regularization_losses
_	variables
`trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
bregularization_losses
c	variables
dtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2Dense_2/kernel
:?2Dense_2/bias
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
hregularization_losses
i	variables
jtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
?
?non_trainable_variables
?layers
 ?layer_regularization_losses
?metrics
nregularization_losses
o	variables
ptrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "RootMeanSquaredError", "name": "root_mean_squared_error", "dtype": "float32", "config": {"name": "root_mean_squared_error", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "c_index", "dtype": "float32", "config": {"name": "c_index", "dtype": "float32", "fn": "c_index"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
':%@2Prot_CNN_0/kernel/m
:@2Prot_CNN_0/bias/m
):'@2SMILES_CNN_0/kernel/m
:@2SMILES_CNN_0/bias/m
':%@@2Prot_CNN_1/kernel/m
:@2Prot_CNN_1/bias/m
):'@@2SMILES_CNN_1/kernel/m
:@2SMILES_CNN_1/bias/m
(:&@?2Prot_CNN_2/kernel/m
:?2Prot_CNN_2/bias/m
*:(@?2SMILES_CNN_2/kernel/m
 :?2SMILES_CNN_2/bias/m
": 
??2Dense_0/kernel/m
:?2Dense_0/bias/m
": 
??2Dense_1/kernel/m
:?2Dense_1/bias/m
": 
??2Dense_2/kernel/m
:?2Dense_2/bias/m
:	?2dense/kernel/m
:2dense/bias/m
':%@2Prot_CNN_0/kernel/v
:@2Prot_CNN_0/bias/v
):'@2SMILES_CNN_0/kernel/v
:@2SMILES_CNN_0/bias/v
':%@@2Prot_CNN_1/kernel/v
:@2Prot_CNN_1/bias/v
):'@@2SMILES_CNN_1/kernel/v
:@2SMILES_CNN_1/bias/v
(:&@?2Prot_CNN_2/kernel/v
:?2Prot_CNN_2/bias/v
*:(@?2SMILES_CNN_2/kernel/v
 :?2SMILES_CNN_2/bias/v
": 
??2Dense_0/kernel/v
:?2Dense_0/bias/v
": 
??2Dense_1/kernel/v
:?2Dense_1/bias/v
": 
??2Dense_2/kernel/v
:?2Dense_2/bias/v
:	?2dense/kernel/v
:2dense/bias/v
?2?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_3471
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_3343
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_2874
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_2936?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_CNN_FCNN_Model_layer_call_fn_3045
-__inference_CNN_FCNN_Model_layer_call_fn_3517
-__inference_CNN_FCNN_Model_layer_call_fn_3153
-__inference_CNN_FCNN_Model_layer_call_fn_3563?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_2391?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *Z?W
U?R
(?%
Protein_Input??????????
	
&?#
SMILES_Input?????????H	
?2?
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_3587
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_3575?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_Prot_Emb_layer_call_fn_3592
'__inference_Prot_Emb_layer_call_fn_3597?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_3621
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_3609?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_SMILES_Emb_layer_call_fn_3626
)__inference_SMILES_Emb_layer_call_fn_3631?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_3647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Prot_CNN_0_layer_call_fn_3656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_3672?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_SMILES_CNN_0_layer_call_fn_3681?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_3697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Prot_CNN_1_layer_call_fn_3706?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_3722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_SMILES_CNN_1_layer_call_fn_3731?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_3747?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Prot_CNN_2_layer_call_fn_3756?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_3772?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_SMILES_CNN_2_layer_call_fn_3781?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_Prot_Global_Max_layer_call_and_return_conditional_losses_2398?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
.__inference_Prot_Global_Max_layer_call_fn_2404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
K__inference_SMILES_Global_Max_layer_call_and_return_conditional_losses_2411?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
0__inference_SMILES_Global_Max_layer_call_fn_2417?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
E__inference_Concatenate_layer_call_and_return_conditional_losses_3788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_Concatenate_layer_call_fn_3794?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_Dense_0_layer_call_and_return_conditional_losses_3805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Dense_0_layer_call_fn_3814?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Dropout_0_layer_call_and_return_conditional_losses_3826
C__inference_Dropout_0_layer_call_and_return_conditional_losses_3831?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_Dropout_0_layer_call_fn_3841
(__inference_Dropout_0_layer_call_fn_3836?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_Dense_1_layer_call_and_return_conditional_losses_3852?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Dense_1_layer_call_fn_3861?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Dropout_1_layer_call_and_return_conditional_losses_3878
C__inference_Dropout_1_layer_call_and_return_conditional_losses_3873?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_Dropout_1_layer_call_fn_3883
(__inference_Dropout_1_layer_call_fn_3888?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_Dense_2_layer_call_and_return_conditional_losses_3899?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Dense_2_layer_call_fn_3908?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_3918?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_3927?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_3201Protein_InputSMILES_Input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_2874?()"#45./@A:;RS\]fglml?i
b?_
U?R
(?%
Protein_Input??????????
	
&?#
SMILES_Input?????????H	
p

 
? "%?"
?
0?????????
? ?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_2936?()"#45./@A:;RS\]fglml?i
b?_
U?R
(?%
Protein_Input??????????
	
&?#
SMILES_Input?????????H	
p 

 
? "%?"
?
0?????????
? ?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_3343?()"#45./@A:;RS\]fglmc?`
Y?V
L?I
#? 
inputs/0??????????
	
"?
inputs/1?????????H	
p

 
? "%?"
?
0?????????
? ?
H__inference_CNN_FCNN_Model_layer_call_and_return_conditional_losses_3471?()"#45./@A:;RS\]fglmc?`
Y?V
L?I
#? 
inputs/0??????????
	
"?
inputs/1?????????H	
p 

 
? "%?"
?
0?????????
? ?
-__inference_CNN_FCNN_Model_layer_call_fn_3045?()"#45./@A:;RS\]fglml?i
b?_
U?R
(?%
Protein_Input??????????
	
&?#
SMILES_Input?????????H	
p

 
? "???????????
-__inference_CNN_FCNN_Model_layer_call_fn_3153?()"#45./@A:;RS\]fglml?i
b?_
U?R
(?%
Protein_Input??????????
	
&?#
SMILES_Input?????????H	
p 

 
? "???????????
-__inference_CNN_FCNN_Model_layer_call_fn_3517?()"#45./@A:;RS\]fglmc?`
Y?V
L?I
#? 
inputs/0??????????
	
"?
inputs/1?????????H	
p

 
? "???????????
-__inference_CNN_FCNN_Model_layer_call_fn_3563?()"#45./@A:;RS\]fglmc?`
Y?V
L?I
#? 
inputs/0??????????
	
"?
inputs/1?????????H	
p 

 
? "???????????
E__inference_Concatenate_layer_call_and_return_conditional_losses_3788?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
*__inference_Concatenate_layer_call_fn_3794y\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "????????????
A__inference_Dense_0_layer_call_and_return_conditional_losses_3805^RS0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_Dense_0_layer_call_fn_3814QRS0?-
&?#
!?
inputs??????????
? "????????????
A__inference_Dense_1_layer_call_and_return_conditional_losses_3852^\]0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_Dense_1_layer_call_fn_3861Q\]0?-
&?#
!?
inputs??????????
? "????????????
A__inference_Dense_2_layer_call_and_return_conditional_losses_3899^fg0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_Dense_2_layer_call_fn_3908Qfg0?-
&?#
!?
inputs??????????
? "????????????
C__inference_Dropout_0_layer_call_and_return_conditional_losses_3826^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
C__inference_Dropout_0_layer_call_and_return_conditional_losses_3831^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? }
(__inference_Dropout_0_layer_call_fn_3836Q4?1
*?'
!?
inputs??????????
p
? "???????????}
(__inference_Dropout_0_layer_call_fn_3841Q4?1
*?'
!?
inputs??????????
p 
? "????????????
C__inference_Dropout_1_layer_call_and_return_conditional_losses_3873^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
C__inference_Dropout_1_layer_call_and_return_conditional_losses_3878^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? }
(__inference_Dropout_1_layer_call_fn_3883Q4?1
*?'
!?
inputs??????????
p
? "???????????}
(__inference_Dropout_1_layer_call_fn_3888Q4?1
*?'
!?
inputs??????????
p 
? "????????????
D__inference_Prot_CNN_0_layer_call_and_return_conditional_losses_3647f"#4?1
*?'
%?"
inputs??????????

? "*?'
 ?
0??????????
@
? ?
)__inference_Prot_CNN_0_layer_call_fn_3656Y"#4?1
*?'
%?"
inputs??????????

? "???????????
@?
D__inference_Prot_CNN_1_layer_call_and_return_conditional_losses_3697f./4?1
*?'
%?"
inputs??????????
@
? "*?'
 ?
0??????????
@
? ?
)__inference_Prot_CNN_1_layer_call_fn_3706Y./4?1
*?'
%?"
inputs??????????
@
? "???????????
@?
D__inference_Prot_CNN_2_layer_call_and_return_conditional_losses_3747g:;4?1
*?'
%?"
inputs??????????
@
? "+?(
!?
0??????????
?
? ?
)__inference_Prot_CNN_2_layer_call_fn_3756Z:;4?1
*?'
%?"
inputs??????????
@
? "???????????
??
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_3575f8?5
.?+
!?
inputs??????????
	

 
p
? "*?'
 ?
0??????????

? ?
B__inference_Prot_Emb_layer_call_and_return_conditional_losses_3587f8?5
.?+
!?
inputs??????????
	

 
p 
? "*?'
 ?
0??????????

? ?
'__inference_Prot_Emb_layer_call_fn_3592Y8?5
.?+
!?
inputs??????????
	

 
p
? "???????????
?
'__inference_Prot_Emb_layer_call_fn_3597Y8?5
.?+
!?
inputs??????????
	

 
p 
? "???????????
?
I__inference_Prot_Global_Max_layer_call_and_return_conditional_losses_2398wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+
$?!
0??????????????????
? ?
.__inference_Prot_Global_Max_layer_call_fn_2404jE?B
;?8
6?3
inputs'???????????????????????????
? "!????????????????????
F__inference_SMILES_CNN_0_layer_call_and_return_conditional_losses_3672d()3?0
)?&
$?!
inputs?????????H
? ")?&
?
0?????????H@
? ?
+__inference_SMILES_CNN_0_layer_call_fn_3681W()3?0
)?&
$?!
inputs?????????H
? "??????????H@?
F__inference_SMILES_CNN_1_layer_call_and_return_conditional_losses_3722d453?0
)?&
$?!
inputs?????????H@
? ")?&
?
0?????????H@
? ?
+__inference_SMILES_CNN_1_layer_call_fn_3731W453?0
)?&
$?!
inputs?????????H@
? "??????????H@?
F__inference_SMILES_CNN_2_layer_call_and_return_conditional_losses_3772e@A3?0
)?&
$?!
inputs?????????H@
? "*?'
 ?
0?????????H?
? ?
+__inference_SMILES_CNN_2_layer_call_fn_3781X@A3?0
)?&
$?!
inputs?????????H@
? "??????????H??
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_3609d7?4
-?*
 ?
inputs?????????H	

 
p
? ")?&
?
0?????????H
? ?
D__inference_SMILES_Emb_layer_call_and_return_conditional_losses_3621d7?4
-?*
 ?
inputs?????????H	

 
p 
? ")?&
?
0?????????H
? ?
)__inference_SMILES_Emb_layer_call_fn_3626W7?4
-?*
 ?
inputs?????????H	

 
p
? "??????????H?
)__inference_SMILES_Emb_layer_call_fn_3631W7?4
-?*
 ?
inputs?????????H	

 
p 
? "??????????H?
K__inference_SMILES_Global_Max_layer_call_and_return_conditional_losses_2411wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+
$?!
0??????????????????
? ?
0__inference_SMILES_Global_Max_layer_call_fn_2417jE?B
;?8
6?3
inputs'???????????????????????????
? "!????????????????????
__inference__wrapped_model_2391?()"#45./@A:;RS\]fglmd?a
Z?W
U?R
(?%
Protein_Input??????????
	
&?#
SMILES_Input?????????H	
? "-?*
(
dense?
dense??????????
?__inference_dense_layer_call_and_return_conditional_losses_3918]lm0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? x
$__inference_dense_layer_call_fn_3927Plm0?-
&?#
!?
inputs??????????
? "???????????
"__inference_signature_wrapper_3201?()"#45./@A:;RS\]fglm??}
? 
v?s
9
Protein_Input(?%
Protein_Input??????????
	
6
SMILES_Input&?#
SMILES_Input?????????H	"-?*
(
dense?
dense?????????