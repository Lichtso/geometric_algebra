use crate::algebra::MultiVectorClass;

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum DataType<'a> {
    Integer,
    SimdVector(usize),
    MultiVector(&'a MultiVectorClass),
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum ExpressionContent<'a> {
    None,
    Variable(&'static str),
    InvokeClassMethod(&'a MultiVectorClass, &'static str, Vec<(DataType<'a>, Expression<'a>)>),
    InvokeInstanceMethod(DataType<'a>, Box<Expression<'a>>, &'static str, Vec<(DataType<'a>, Expression<'a>)>),
    Conversion(&'a MultiVectorClass, &'a MultiVectorClass, Box<Expression<'a>>),
    Select(Box<Expression<'a>>, Box<Expression<'a>>, Box<Expression<'a>>),
    Access(Box<Expression<'a>>, usize),
    Swizzle(Box<Expression<'a>>, Vec<usize>),
    Gather(Box<Expression<'a>>, Vec<(usize, usize)>),
    Constant(DataType<'a>, Vec<isize>),
    SquareRoot(Box<Expression<'a>>),
    Add(Box<Expression<'a>>, Box<Expression<'a>>),
    Subtract(Box<Expression<'a>>, Box<Expression<'a>>),
    Multiply(Box<Expression<'a>>, Box<Expression<'a>>),
    Divide(Box<Expression<'a>>, Box<Expression<'a>>),
    LessThan(Box<Expression<'a>>, Box<Expression<'a>>),
    Equal(Box<Expression<'a>>, Box<Expression<'a>>),
    LogicAnd(Box<Expression<'a>>, Box<Expression<'a>>),
    BitShiftRight(Box<Expression<'a>>, Box<Expression<'a>>),
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Expression<'a> {
    pub size: usize,
    pub content: ExpressionContent<'a>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Parameter<'a> {
    pub name: &'static str,
    pub data_type: DataType<'a>,
}

impl<'a> Parameter<'a> {
    pub fn multi_vector_class(&self) -> &'a MultiVectorClass {
        if let DataType::MultiVector(class) = self.data_type {
            class
        } else {
            unreachable!()
        }
    }
}

#[derive(PartialEq, Eq, Clone)]
pub enum AstNode<'a> {
    None,
    Preamble,
    ClassDefinition {
        class: &'a MultiVectorClass,
    },
    ReturnStatement {
        expression: Box<Expression<'a>>,
    },
    VariableAssignment {
        name: &'static str,
        data_type: Option<DataType<'a>>,
        expression: Box<Expression<'a>>,
    },
    IfThenBlock {
        condition: Box<Expression<'a>>,
        body: Vec<AstNode<'a>>,
    },
    WhileLoopBlock {
        condition: Box<Expression<'a>>,
        body: Vec<AstNode<'a>>,
    },
    TraitImplementation {
        result: Parameter<'a>,
        parameters: Vec<Parameter<'a>>,
        body: Vec<AstNode<'a>>,
    },
}
