use crate::{
    algebra::{BasisElement, BasisElementIndex, Involution, MultiVectorClass, MultiVectorClassRegistry, Product},
    ast::{AstNode, DataType, Expression, ExpressionContent, Parameter},
};

#[macro_export]
macro_rules! result_of_trait {
    ($ast_node:expr) => {
        match $ast_node {
            AstNode::TraitImplementation { ref result, .. } => result,
            _ => unreachable!(),
        }
    };
}

pub fn simplify_and_legalize(expression: Box<Expression>) -> Box<Expression> {
    match expression.content {
        ExpressionContent::Gather(mut inner_expression, indices) => {
            if let Some(first_index_pair) = indices.first() {
                inner_expression = simplify_and_legalize(inner_expression);
                if indices.iter().all(|index_pair| index_pair == first_index_pair) {
                    Box::new(Expression {
                        size: expression.size,
                        content: ExpressionContent::Gather(inner_expression, vec![*first_index_pair]),
                    })
                } else if inner_expression.size == expression.size && indices.iter().all(|(array_index, _)| *array_index == first_index_pair.0) {
                    inner_expression = Box::new(Expression {
                        size: expression.size,
                        content: ExpressionContent::Access(inner_expression, first_index_pair.0),
                    });
                    if indices.iter().enumerate().any(|(i, (_, component_index))| i != *component_index) {
                        Box::new(Expression {
                            size: expression.size,
                            content: ExpressionContent::Swizzle(
                                inner_expression,
                                indices.iter().map(|(_, component_index)| *component_index).collect(),
                            ),
                        })
                    } else {
                        inner_expression
                    }
                } else {
                    Box::new(Expression {
                        size: expression.size,
                        content: ExpressionContent::Gather(inner_expression, indices),
                    })
                }
            } else {
                Box::new(Expression {
                    size: expression.size,
                    content: ExpressionContent::None,
                })
            }
        }
        ExpressionContent::Constant(ref data_type, ref values) => {
            let first_value = values.first().unwrap();
            if values.iter().all(|value| value == first_value) {
                Box::new(Expression {
                    size: expression.size,
                    content: ExpressionContent::Constant(data_type.clone(), vec![*first_value]),
                })
            } else {
                expression
            }
        }
        ExpressionContent::Add(mut a, mut b) => {
            if let ExpressionContent::Multiply(ref c, ref d) = b.content {
                if let ExpressionContent::Multiply(ref e, ref f) = d.content {
                    if let ExpressionContent::Constant(_data_type, values) = &f.content {
                        if values.iter().all(|value| *value == -1) {
                            b = Box::new(Expression {
                                size: expression.size,
                                content: ExpressionContent::Multiply(c.clone(), e.clone()),
                            });
                            return simplify_and_legalize(Box::new(Expression {
                                size: expression.size,
                                content: ExpressionContent::Subtract(a, b),
                            }));
                        }
                    }
                }
            }
            a = simplify_and_legalize(a);
            b = simplify_and_legalize(b);
            if a.content == ExpressionContent::None {
                b
            } else if b.content == ExpressionContent::None {
                a
            } else {
                Box::new(Expression {
                    size: expression.size,
                    content: ExpressionContent::Add(a, b),
                })
            }
        }
        ExpressionContent::Subtract(mut a, mut b) => {
            a = simplify_and_legalize(a);
            b = simplify_and_legalize(b);
            if a.content == ExpressionContent::None {
                let constant = Expression {
                    size: expression.size,
                    content: ExpressionContent::Constant(DataType::SimdVector(expression.size), vec![0]),
                };
                Box::new(Expression {
                    size: expression.size,
                    content: ExpressionContent::Subtract(Box::new(constant), b),
                })
            } else if b.content == ExpressionContent::None {
                a
            } else {
                Box::new(Expression {
                    size: expression.size,
                    content: ExpressionContent::Subtract(a, b),
                })
            }
        }
        ExpressionContent::Multiply(mut a, mut b) => {
            a = simplify_and_legalize(a);
            b = simplify_and_legalize(b);
            if let ExpressionContent::Constant(_, _) = a.content {
                std::mem::swap(&mut a, &mut b)
            }
            if a.content == ExpressionContent::None {
                b
            } else {
                match b.content {
                    ExpressionContent::None => a,
                    ExpressionContent::Constant(_data_type, c) if c.iter().all(|c| *c == 1) => a,
                    ExpressionContent::Constant(_data_type, c) if c.iter().all(|c| *c == 0) => Box::new(Expression {
                        size: expression.size,
                        content: ExpressionContent::None,
                    }),
                    _ => Box::new(Expression {
                        size: expression.size,
                        content: ExpressionContent::Multiply(a, b),
                    }),
                }
            }
        }
        _ => expression,
    }
}

impl MultiVectorClass {
    pub fn flat_basis(&self) -> Vec<BasisElement> {
        self.grouped_basis.iter().flatten().cloned().collect()
    }

    pub fn signature(&self) -> Vec<BasisElementIndex> {
        let mut signature: Vec<BasisElementIndex> = self.grouped_basis.iter().flatten().map(|element| element.index).collect();
        signature.sort_unstable();
        signature
    }

    pub fn index_in_group(&self, mut index: usize) -> (usize, usize) {
        for (group_index, group) in self.grouped_basis.iter().enumerate() {
            if index >= group.len() {
                index -= group.len();
            } else {
                return (group_index, index);
            }
        }
        unreachable!()
    }

    pub fn constant<'a>(&'a self, name: &'static str) -> AstNode<'a> {
        let (scalar_value, other_values) = match name {
            "Zero" => (0, 0),
            "One" => (1, 0),
            _ => unreachable!(),
        };
        let mut body = Vec::new();
        for result_group in self.grouped_basis.iter() {
            let size = result_group.len();
            let expression = Expression {
                size,
                content: ExpressionContent::Constant(
                    DataType::SimdVector(size),
                    result_group
                        .iter()
                        .map(|element| if element.index == 0 { scalar_value } else { other_values })
                        .collect(),
                ),
            };
            body.push((DataType::SimdVector(size), *simplify_and_legalize(Box::new(expression))));
        }
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: DataType::MultiVector(self),
            },
            parameters: vec![],
            body: vec![AstNode::ReturnStatement {
                expression: Box::new(Expression {
                    size: 1,
                    content: ExpressionContent::InvokeClassMethod(self, "Constructor", body),
                }),
            }],
        }
    }

    pub fn involution<'a>(
        name: &'static str,
        involution: &Involution,
        parameter_a: &Parameter<'a>,
        registry: &'a MultiVectorClassRegistry,
        project: bool,
    ) -> AstNode<'a> {
        let a_flat_basis = parameter_a.multi_vector_class().flat_basis();
        let mut result_signature = Vec::new();
        for a_element in a_flat_basis.iter() {
            for (in_element, out_element) in involution.terms.iter() {
                if in_element.index == a_element.index {
                    result_signature.push(out_element.index);
                    break;
                }
            }
        }
        if project {
            for (in_element, _out_element) in involution.terms.iter() {
                if !a_flat_basis.iter().any(|element| element.index == in_element.index) {
                    return AstNode::None;
                }
            }
        }
        result_signature.sort_unstable();
        if let Some(result_class) = registry.get(&result_signature) {
            let result_flat_basis = result_class.flat_basis();
            let mut body = Vec::new();
            let mut base_index = 0;
            for result_group in result_class.grouped_basis.iter() {
                let size = result_group.len();
                let (factors, a_indices): (Vec<_>, Vec<_>) = (0..size)
                    .map(|index_in_group| {
                        let result_element = &result_flat_basis[base_index + index_in_group];
                        let involution_element = involution
                            .terms
                            .iter()
                            .position(|(_in_element, out_element)| out_element.index == result_element.index)
                            .unwrap();
                        let (in_element, out_element) = &involution.terms[involution_element];
                        let index_in_a = a_flat_basis.iter().position(|a_element| a_element.index == in_element.index).unwrap();
                        (
                            out_element.scalar * result_element.scalar * in_element.scalar * a_flat_basis[index_in_a].scalar,
                            parameter_a.multi_vector_class().index_in_group(index_in_a),
                        )
                    })
                    .unzip();
                let a_group_index = a_indices[0].0;
                let expression = Expression {
                    size,
                    content: ExpressionContent::Multiply(
                        Box::new(Expression {
                            size,
                            content: ExpressionContent::Gather(
                                Box::new(Expression {
                                    size: parameter_a.multi_vector_class().grouped_basis[a_group_index].len(),
                                    content: ExpressionContent::Variable(parameter_a.name),
                                }),
                                a_indices,
                            ),
                        }),
                        Box::new(Expression {
                            size,
                            content: ExpressionContent::Constant(DataType::SimdVector(size), factors),
                        }),
                    ),
                };
                body.push((DataType::SimdVector(size), *simplify_and_legalize(Box::new(expression))));
                base_index += size;
            }
            AstNode::TraitImplementation {
                result: Parameter {
                    name,
                    data_type: DataType::MultiVector(result_class),
                },
                parameters: vec![parameter_a.clone()],
                body: vec![AstNode::ReturnStatement {
                    expression: Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::InvokeClassMethod(result_class, "Constructor", body),
                    }),
                }],
            }
        } else {
            AstNode::None
        }
    }

    pub fn element_wise<'a>(
        name: &'static str,
        parameter_a: &Parameter<'a>,
        parameter_b: &Parameter<'a>,
        registry: &'a MultiVectorClassRegistry,
    ) -> AstNode<'a> {
        let a_flat_basis = parameter_a.multi_vector_class().flat_basis();
        let b_flat_basis = parameter_b.multi_vector_class().flat_basis();
        let result_signature = a_flat_basis
            .iter()
            .chain(b_flat_basis.iter())
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let mut result_signature = result_signature.into_iter().map(|element| element.index).collect::<Vec<_>>();
        result_signature.sort_unstable();
        if let Some(result_class) = registry.get(&result_signature) {
            let parameters = [(parameter_a, &a_flat_basis), (parameter_b, &b_flat_basis)];
            let mut body = Vec::new();
            for result_group in result_class.grouped_basis.iter() {
                let size = result_group.len();
                let mut expressions = parameters.iter().map(|(parameter, flat_basis)| {
                    let mut parameter_group_index = None;
                    let terms: Vec<_> = result_group
                        .iter()
                        .map(|result_element| {
                            if let Some(index_in_flat_basis) = flat_basis.iter().position(|element| element.index == result_element.index) {
                                let index_pair = parameter.multi_vector_class().index_in_group(index_in_flat_basis);
                                parameter_group_index = Some(index_pair.0);
                                (result_element.scalar * flat_basis[index_in_flat_basis].scalar, index_pair)
                            } else {
                                (0, (0, 0))
                            }
                        })
                        .collect();
                    Expression {
                        size,
                        content: ExpressionContent::Multiply(
                            Box::new(Expression {
                                size,
                                content: ExpressionContent::Gather(
                                    Box::new(Expression {
                                        size: if let Some(index) = parameter_group_index {
                                            parameter.multi_vector_class().grouped_basis[index].len()
                                        } else {
                                            size
                                        },
                                        content: ExpressionContent::Variable(parameter.name),
                                    }),
                                    terms.iter().map(|(_factor, index_pair)| index_pair).cloned().collect(),
                                ),
                            }),
                            Box::new(Expression {
                                size,
                                content: ExpressionContent::Constant(
                                    DataType::SimdVector(size),
                                    terms.iter().map(|(factor, _index_pair)| *factor).collect::<Vec<_>>(),
                                ),
                            }),
                        ),
                    }
                });
                body.push((
                    DataType::SimdVector(size),
                    *simplify_and_legalize(Box::new(Expression {
                        size,
                        content: match name {
                            "Add" => ExpressionContent::Add(Box::new(expressions.next().unwrap()), Box::new(expressions.next().unwrap())),
                            "Sub" => ExpressionContent::Subtract(Box::new(expressions.next().unwrap()), Box::new(expressions.next().unwrap())),
                            "Mul" => ExpressionContent::Multiply(Box::new(expressions.next().unwrap()), Box::new(expressions.next().unwrap())),
                            "Div" => ExpressionContent::Divide(Box::new(expressions.next().unwrap()), Box::new(expressions.next().unwrap())),
                            _ => unreachable!(),
                        },
                    })),
                ));
            }
            AstNode::TraitImplementation {
                result: Parameter {
                    name,
                    data_type: DataType::MultiVector(result_class),
                },
                parameters: vec![parameter_a.clone(), parameter_b.clone()],
                body: vec![AstNode::ReturnStatement {
                    expression: Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::InvokeClassMethod(result_class, "Constructor", body),
                    }),
                }],
            }
        } else {
            AstNode::None
        }
    }

    pub fn product<'a>(
        name: &'static str,
        product: &Product,
        parameter_a: &Parameter<'a>,
        parameter_b: &Parameter<'a>,
        registry: &'a MultiVectorClassRegistry,
    ) -> AstNode<'a> {
        let a_flat_basis = parameter_a.multi_vector_class().flat_basis();
        let b_flat_basis = parameter_b.multi_vector_class().flat_basis();
        let mut result_signature = std::collections::HashSet::new();
        for product_term in product.terms.iter() {
            if a_flat_basis.iter().any(|e| e.index == product_term.factor_a.index)
                && b_flat_basis.iter().any(|e| e.index == product_term.factor_b.index)
            {
                result_signature.insert(product_term.product.index);
            }
        }
        let mut result_signature = result_signature.into_iter().collect::<Vec<_>>();
        result_signature.sort_unstable();
        if let Some(result_class) = registry.get(&result_signature) {
            let result_flat_basis = result_class.flat_basis();
            let mut sorted_terms = vec![vec![(0, 0); a_flat_basis.len()]; result_flat_basis.len()];
            for product_term in product.terms.iter() {
                if let Some(y) = result_flat_basis.iter().position(|e| e.index == product_term.product.index) {
                    if let Some(x) = a_flat_basis.iter().position(|e| e.index == product_term.factor_a.index) {
                        if let Some(gather_index) = b_flat_basis.iter().position(|e| e.index == product_term.factor_b.index) {
                            sorted_terms[y][x] = (
                                result_flat_basis[y].scalar
                                    * product_term.product.scalar
                                    * a_flat_basis[x].scalar
                                    * product_term.factor_a.scalar
                                    * b_flat_basis[gather_index].scalar
                                    * product_term.factor_b.scalar,
                                gather_index,
                            );
                        }
                    }
                }
            }
            let mut body = Vec::new();
            let mut base_index = 0;
            for result_group in result_class.grouped_basis.iter() {
                let size = result_group.len();
                let mut expression = Expression {
                    size,
                    content: ExpressionContent::None,
                };
                let result_terms = (0..size)
                    .map(|index_in_group| &sorted_terms[base_index + index_in_group])
                    .collect::<Vec<_>>();
                let transposed_terms = (0..result_terms[0].len()).map(|i| result_terms.iter().map(|inner| inner[i]).collect::<Vec<_>>());
                let mut contraction = (
                    Expression {
                        size,
                        content: ExpressionContent::None,
                    },
                    Expression {
                        size,
                        content: ExpressionContent::None,
                    },
                    vec![(0, 0); expression.size],
                    vec![(0, 0); expression.size],
                    vec![0; expression.size],
                );
                for (index_in_a, a_terms) in transposed_terms.enumerate() {
                    if a_terms.iter().all(|(factor, _)| *factor == 0) {
                        continue;
                    }
                    let (a_group_index, a_index_in_group) = parameter_a.multi_vector_class().index_in_group(index_in_a);
                    let a_indices = a_terms.iter().map(|_| (a_group_index, a_index_in_group)).collect::<Vec<_>>();
                    let b_indices = a_terms
                        .iter()
                        .map(|(_, index_in_b)| parameter_b.multi_vector_class().index_in_group(*index_in_b))
                        .collect::<Vec<_>>();
                    let non_zero_index = a_terms.iter().position(|(factor, _index_pair)| *factor != 0).unwrap();
                    let b_group_index = b_indices[non_zero_index].0;
                    let b_indices = a_terms
                        .iter()
                        .enumerate()
                        .map(|(index, (factor, _index_pair))| b_indices[if *factor == 0 { non_zero_index } else { index }])
                        .collect::<Vec<_>>();
                    let is_contractable = a_terms.iter().enumerate().all(|(i, (factor, _))| *factor == 0 || contraction.4[i] == 0)
                        && (contraction.0.content == ExpressionContent::None
                            || contraction.0.size == parameter_a.multi_vector_class().grouped_basis[a_group_index].len())
                        && (contraction.1.content == ExpressionContent::None
                            || contraction.1.size == parameter_b.multi_vector_class().grouped_basis[b_group_index].len());
                    if is_contractable && a_terms.iter().any(|(factor, _)| *factor == 0) {
                        if contraction.0.content == ExpressionContent::None {
                            assert!(contraction.1.content == ExpressionContent::None);
                            contraction.0 = Expression {
                                size: parameter_a.multi_vector_class().grouped_basis[a_group_index].len(),
                                content: ExpressionContent::Variable(parameter_a.name),
                            };
                            contraction.1 = Expression {
                                size: parameter_b.multi_vector_class().grouped_basis[b_group_index].len(),
                                content: ExpressionContent::Variable(parameter_b.name),
                            };
                            contraction.2 = a_indices.iter().map(|(a_group_index, _)| (*a_group_index, 0)).collect();
                            contraction.3 = b_indices.iter().map(|(b_group_index, _)| (*b_group_index, 0)).collect();
                        }
                        for (i, (factor, _index_in_b)) in a_terms.iter().enumerate() {
                            if *factor != 0 {
                                contraction.2[i] = a_indices[i];
                                contraction.3[i] = b_indices[i];
                                contraction.4[i] = *factor;
                            }
                        }
                    } else {
                        expression = Expression {
                            size,
                            content: ExpressionContent::Add(
                                Box::new(expression),
                                Box::new(Expression {
                                    size,
                                    content: ExpressionContent::Multiply(
                                        Box::new(Expression {
                                            size,
                                            content: ExpressionContent::Gather(
                                                Box::new(Expression {
                                                    size: parameter_a.multi_vector_class().grouped_basis[a_group_index].len(),
                                                    content: ExpressionContent::Variable(parameter_a.name),
                                                }),
                                                a_indices,
                                            ),
                                        }),
                                        Box::new(Expression {
                                            size,
                                            content: ExpressionContent::Multiply(
                                                Box::new(Expression {
                                                    size,
                                                    content: ExpressionContent::Gather(
                                                        Box::new(Expression {
                                                            size: parameter_b.multi_vector_class().grouped_basis[b_group_index].len(),
                                                            content: ExpressionContent::Variable(parameter_b.name),
                                                        }),
                                                        b_indices,
                                                    ),
                                                }),
                                                Box::new(Expression {
                                                    size,
                                                    content: ExpressionContent::Constant(
                                                        DataType::SimdVector(size),
                                                        a_terms.iter().map(|(factor, _)| *factor).collect::<Vec<_>>(),
                                                    ),
                                                }),
                                            ),
                                        }),
                                    ),
                                }),
                            ),
                        };
                    }
                }
                if contraction.4.iter().any(|scalar| *scalar != 0) {
                    expression = Expression {
                        size,
                        content: ExpressionContent::Add(
                            Box::new(expression),
                            Box::new(Expression {
                                size,
                                content: ExpressionContent::Multiply(
                                    Box::new(Expression {
                                        size,
                                        content: ExpressionContent::Multiply(
                                            Box::new(Expression {
                                                size,
                                                content: ExpressionContent::Gather(Box::new(contraction.0), contraction.2),
                                            }),
                                            Box::new(Expression {
                                                size,
                                                content: ExpressionContent::Gather(Box::new(contraction.1), contraction.3),
                                            }),
                                        ),
                                    }),
                                    Box::new(Expression {
                                        size,
                                        content: ExpressionContent::Constant(DataType::SimdVector(size), contraction.4),
                                    }),
                                ),
                            }),
                        ),
                    };
                }
                if expression.content == ExpressionContent::None {
                    expression = Expression {
                        size,
                        content: ExpressionContent::Constant(DataType::SimdVector(size), (0..size).map(|_| 0).collect()),
                    };
                }
                body.push((DataType::SimdVector(size), *simplify_and_legalize(Box::new(expression))));
                base_index += size;
            }
            if body.is_empty() {
                AstNode::None
            } else {
                AstNode::TraitImplementation {
                    result: Parameter {
                        name,
                        data_type: DataType::MultiVector(result_class),
                    },
                    parameters: vec![parameter_a.clone(), parameter_b.clone()],
                    body: vec![AstNode::ReturnStatement {
                        expression: Box::new(Expression {
                            size: 1,
                            content: ExpressionContent::InvokeClassMethod(result_class, "Constructor", body),
                        }),
                    }],
                }
            }
        } else {
            AstNode::None
        }
    }

    pub fn derive_squared_magnitude<'a>(
        name: &'static str,
        scalar_product: &AstNode<'a>,
        involution: &AstNode<'a>,
        parameter_a: &Parameter<'a>,
    ) -> AstNode<'a> {
        let scalar_product_result = result_of_trait!(scalar_product);
        let involution_result = result_of_trait!(involution);
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: scalar_product_result.data_type.clone(),
            },
            parameters: vec![parameter_a.clone()],
            body: vec![AstNode::ReturnStatement {
                expression: Box::new(Expression {
                    size: 1,
                    content: ExpressionContent::InvokeInstanceMethod(
                        parameter_a.data_type.clone(),
                        Box::new(Expression {
                            size: 1,
                            content: ExpressionContent::Variable(parameter_a.name),
                        }),
                        scalar_product_result.name,
                        vec![(
                            DataType::MultiVector(involution_result.multi_vector_class()),
                            Expression {
                                size: 1,
                                content: ExpressionContent::InvokeInstanceMethod(
                                    parameter_a.data_type.clone(),
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Variable(parameter_a.name),
                                    }),
                                    involution_result.name,
                                    vec![],
                                ),
                            },
                        )],
                    ),
                }),
            }],
        }
    }

    pub fn derive_scale<'a>(
        name: &'static str,
        geometric_product: &AstNode<'a>,
        parameter_a: &Parameter<'a>,
        parameter_b: &Parameter<'a>,
    ) -> AstNode<'a> {
        let geometric_product_result = result_of_trait!(geometric_product);
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: geometric_product_result.data_type.clone(),
            },
            parameters: vec![
                parameter_a.clone(),
                Parameter {
                    name: "other",
                    data_type: DataType::SimdVector(1),
                },
            ],
            body: vec![AstNode::ReturnStatement {
                expression: Box::new(Expression {
                    size: 1,
                    content: ExpressionContent::InvokeInstanceMethod(
                        parameter_a.data_type.clone(),
                        Box::new(Expression {
                            size: 1,
                            content: ExpressionContent::Variable(parameter_a.name),
                        }),
                        geometric_product_result.name,
                        vec![(
                            DataType::MultiVector(parameter_b.multi_vector_class()),
                            Expression {
                                size: 1,
                                content: ExpressionContent::InvokeClassMethod(
                                    parameter_b.multi_vector_class(),
                                    "Constructor",
                                    vec![(
                                        DataType::SimdVector(1),
                                        Expression {
                                            size: 1,
                                            content: ExpressionContent::Variable(parameter_b.name),
                                        },
                                    )],
                                ),
                            },
                        )],
                    ),
                }),
            }],
        }
    }

    pub fn derive_magnitude<'a>(name: &'static str, squared_magnitude: &AstNode<'a>, parameter_a: &Parameter<'a>) -> AstNode<'a> {
        let squared_magnitude_result = result_of_trait!(squared_magnitude);
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: squared_magnitude_result.data_type.clone(),
            },
            parameters: vec![parameter_a.clone()],
            body: vec![AstNode::ReturnStatement {
                expression: Box::new(Expression {
                    size: 1,
                    content: ExpressionContent::InvokeClassMethod(
                        squared_magnitude_result.multi_vector_class(),
                        "Constructor",
                        vec![(
                            DataType::SimdVector(1),
                            Expression {
                                size: 1,
                                content: ExpressionContent::SquareRoot(Box::new(Expression {
                                    size: 1,
                                    content: ExpressionContent::Access(
                                        Box::new(Expression {
                                            size: 1,
                                            content: ExpressionContent::InvokeInstanceMethod(
                                                parameter_a.data_type.clone(),
                                                Box::new(Expression {
                                                    size: 1,
                                                    content: ExpressionContent::Variable(parameter_a.name),
                                                }),
                                                squared_magnitude_result.name,
                                                vec![],
                                            ),
                                        }),
                                        0,
                                    ),
                                })),
                            },
                        )],
                    ),
                }),
            }],
        }
    }

    pub fn derive_signum<'a>(
        name: &'static str,
        geometric_product: &AstNode<'a>,
        magnitude: &AstNode<'a>,
        parameter_a: &Parameter<'a>,
    ) -> AstNode<'a> {
        let geometric_product_result = result_of_trait!(geometric_product);
        let magnitude_result = result_of_trait!(magnitude);
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: geometric_product_result.data_type.clone(),
            },
            parameters: vec![parameter_a.clone()],
            body: vec![AstNode::ReturnStatement {
                expression: Box::new(Expression {
                    size: 1,
                    content: ExpressionContent::InvokeInstanceMethod(
                        parameter_a.data_type.clone(),
                        Box::new(Expression {
                            size: 1,
                            content: ExpressionContent::Variable(parameter_a.name),
                        }),
                        geometric_product_result.name,
                        vec![(
                            DataType::MultiVector(magnitude_result.multi_vector_class()),
                            Expression {
                                size: 1,
                                content: ExpressionContent::InvokeClassMethod(
                                    magnitude_result.multi_vector_class(),
                                    "Constructor",
                                    vec![(
                                        DataType::SimdVector(1),
                                        Expression {
                                            size: 1,
                                            content: ExpressionContent::Divide(
                                                Box::new(Expression {
                                                    size: 1,
                                                    content: ExpressionContent::Constant(DataType::SimdVector(1), vec![1]),
                                                }),
                                                Box::new(Expression {
                                                    size: 1,
                                                    content: ExpressionContent::Access(
                                                        Box::new(Expression {
                                                            size: 1,
                                                            content: ExpressionContent::InvokeInstanceMethod(
                                                                parameter_a.data_type.clone(),
                                                                Box::new(Expression {
                                                                    size: 1,
                                                                    content: ExpressionContent::Variable(parameter_a.name),
                                                                }),
                                                                magnitude_result.name,
                                                                vec![],
                                                            ),
                                                        }),
                                                        0,
                                                    ),
                                                }),
                                            ),
                                        },
                                    )],
                                ),
                            },
                        )],
                    ),
                }),
            }],
        }
    }

    pub fn derive_inverse<'a>(
        name: &'static str,
        geometric_product: &AstNode<'a>,
        squared_magnitude: &AstNode<'a>,
        involution: &AstNode<'a>,
        parameter_a: &Parameter<'a>,
    ) -> AstNode<'a> {
        let geometric_product_result = result_of_trait!(geometric_product);
        let squared_magnitude_result = result_of_trait!(squared_magnitude);
        let involution_result = result_of_trait!(involution);
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: geometric_product_result.data_type.clone(),
            },
            parameters: vec![parameter_a.clone()],
            body: vec![AstNode::ReturnStatement {
                expression: Box::new(Expression {
                    size: 1,
                    content: ExpressionContent::InvokeInstanceMethod(
                        involution_result.data_type.clone(),
                        Box::new(Expression {
                            size: 1,
                            content: ExpressionContent::InvokeInstanceMethod(
                                parameter_a.data_type.clone(),
                                Box::new(Expression {
                                    size: 1,
                                    content: ExpressionContent::Variable(parameter_a.name),
                                }),
                                involution_result.name,
                                vec![],
                            ),
                        }),
                        geometric_product_result.name,
                        vec![(
                            DataType::MultiVector(squared_magnitude_result.multi_vector_class()),
                            Expression {
                                size: 1,
                                content: ExpressionContent::InvokeClassMethod(
                                    squared_magnitude_result.multi_vector_class(),
                                    "Constructor",
                                    vec![(
                                        DataType::SimdVector(1),
                                        Expression {
                                            size: 1,
                                            content: ExpressionContent::Divide(
                                                Box::new(Expression {
                                                    size: 1,
                                                    content: ExpressionContent::Constant(DataType::SimdVector(1), vec![1]),
                                                }),
                                                Box::new(Expression {
                                                    size: 1,
                                                    content: ExpressionContent::Access(
                                                        Box::new(Expression {
                                                            size: 1,
                                                            content: ExpressionContent::InvokeInstanceMethod(
                                                                parameter_a.data_type.clone(),
                                                                Box::new(Expression {
                                                                    size: 1,
                                                                    content: ExpressionContent::Variable(parameter_a.name),
                                                                }),
                                                                squared_magnitude_result.name,
                                                                vec![],
                                                            ),
                                                        }),
                                                        0,
                                                    ),
                                                }),
                                            ),
                                        },
                                    )],
                                ),
                            },
                        )],
                    ),
                }),
            }],
        }
    }

    pub fn derive_power_of_integer<'a>(
        name: &'static str,
        geometric_product: &AstNode<'a>,
        constant_one: &AstNode<'a>,
        inverse: &AstNode<'a>,
        parameter_a: &Parameter<'a>,
        parameter_b: &Parameter<'a>,
    ) -> AstNode<'a> {
        let geometric_product_result = result_of_trait!(geometric_product);
        let constant_one_result = result_of_trait!(constant_one);
        let inverse_result = result_of_trait!(inverse);
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: parameter_a.data_type.clone(),
            },
            parameters: vec![parameter_a.clone(), parameter_b.clone()],
            body: vec![
                AstNode::IfThenBlock {
                    condition: Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::Equal(
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Variable(parameter_b.name),
                            }),
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Constant(DataType::Integer, vec![0]),
                            }),
                        ),
                    }),
                    body: vec![AstNode::ReturnStatement {
                        expression: Box::new(Expression {
                            size: 1,
                            content: ExpressionContent::InvokeClassMethod(parameter_a.multi_vector_class(), constant_one_result.name, vec![]),
                        }),
                    }],
                },
                AstNode::VariableAssignment {
                    name: "x",
                    data_type: Some(parameter_a.data_type.clone()),
                    expression: Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::Select(
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::LessThan(
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Variable(parameter_b.name),
                                    }),
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Constant(DataType::Integer, vec![0]),
                                    }),
                                ),
                            }),
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::InvokeInstanceMethod(
                                    parameter_a.data_type.clone(),
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Variable(parameter_a.name),
                                    }),
                                    inverse_result.name,
                                    vec![],
                                ),
                            }),
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Variable(parameter_a.name),
                            }),
                        ),
                    }),
                },
                AstNode::VariableAssignment {
                    name: "y",
                    data_type: Some(parameter_a.data_type.clone()),
                    expression: Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::InvokeClassMethod(parameter_a.multi_vector_class(), constant_one_result.name, vec![]),
                    }),
                },
                AstNode::VariableAssignment {
                    name: "n",
                    data_type: Some(DataType::Integer),
                    expression: Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::InvokeInstanceMethod(
                            DataType::Integer,
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Variable(parameter_b.name),
                            }),
                            "Abs",
                            vec![],
                        ),
                    }),
                },
                AstNode::WhileLoopBlock {
                    condition: Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::LessThan(
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Constant(DataType::Integer, vec![1]),
                            }),
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Variable("n"),
                            }),
                        ),
                    }),
                    body: vec![
                        AstNode::IfThenBlock {
                            condition: Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Equal(
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::LogicAnd(
                                            Box::new(Expression {
                                                size: 1,
                                                content: ExpressionContent::Variable("n"),
                                            }),
                                            Box::new(Expression {
                                                size: 1,
                                                content: ExpressionContent::Constant(DataType::Integer, vec![1]),
                                            }),
                                        ),
                                    }),
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Constant(DataType::Integer, vec![1]),
                                    }),
                                ),
                            }),
                            body: vec![AstNode::VariableAssignment {
                                name: "y",
                                data_type: None,
                                expression: Box::new(Expression {
                                    size: 1,
                                    content: ExpressionContent::InvokeInstanceMethod(
                                        parameter_a.data_type.clone(),
                                        Box::new(Expression {
                                            size: 1,
                                            content: ExpressionContent::Variable("x"),
                                        }),
                                        geometric_product_result.name,
                                        vec![(
                                            DataType::MultiVector(parameter_a.multi_vector_class()),
                                            Expression {
                                                size: 1,
                                                content: ExpressionContent::Variable("y"),
                                            },
                                        )],
                                    ),
                                }),
                            }],
                        },
                        AstNode::VariableAssignment {
                            name: "x",
                            data_type: None,
                            expression: Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::InvokeInstanceMethod(
                                    parameter_a.data_type.clone(),
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Variable("x"),
                                    }),
                                    geometric_product_result.name,
                                    vec![(
                                        DataType::MultiVector(parameter_a.multi_vector_class()),
                                        Expression {
                                            size: 1,
                                            content: ExpressionContent::Variable("x"),
                                        },
                                    )],
                                ),
                            }),
                        },
                        AstNode::VariableAssignment {
                            name: "n",
                            data_type: None,
                            expression: Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::BitShiftRight(
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Variable("n"),
                                    }),
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Constant(DataType::Integer, vec![1]),
                                    }),
                                ),
                            }),
                        },
                    ],
                },
                AstNode::ReturnStatement {
                    expression: Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::InvokeInstanceMethod(
                            parameter_a.data_type.clone(),
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Variable("x"),
                            }),
                            geometric_product_result.name,
                            vec![(
                                DataType::MultiVector(parameter_a.multi_vector_class()),
                                Expression {
                                    size: 1,
                                    content: ExpressionContent::Variable("y"),
                                },
                            )],
                        ),
                    }),
                },
            ],
        }
    }

    pub fn derive_division<'a>(
        name: &'static str,
        geometric_product: &AstNode<'a>,
        inverse: &AstNode<'a>,
        parameter_a: &Parameter<'a>,
        parameter_b: &Parameter<'a>,
    ) -> AstNode<'a> {
        let geometric_product_result = result_of_trait!(geometric_product);
        let inverse_result = result_of_trait!(inverse);
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: geometric_product_result.data_type.clone(),
            },
            parameters: vec![parameter_a.clone(), parameter_b.clone()],
            body: vec![AstNode::ReturnStatement {
                expression: Box::new(Expression {
                    size: 1,
                    content: ExpressionContent::InvokeInstanceMethod(
                        parameter_a.data_type.clone(),
                        Box::new(Expression {
                            size: 1,
                            content: ExpressionContent::Variable(parameter_a.name),
                        }),
                        geometric_product_result.name,
                        vec![(
                            DataType::MultiVector(inverse_result.multi_vector_class()),
                            Expression {
                                size: 1,
                                content: ExpressionContent::InvokeInstanceMethod(
                                    parameter_b.data_type.clone(),
                                    Box::new(Expression {
                                        size: 1,
                                        content: ExpressionContent::Variable(parameter_b.name),
                                    }),
                                    inverse_result.name,
                                    vec![],
                                ),
                            },
                        )],
                    ),
                }),
            }],
        }
    }

    pub fn derive_sandwich_product<'a>(
        name: &'static str,
        geometric_product: &AstNode<'a>,
        geometric_product_2: &AstNode<'a>,
        involution: &AstNode<'a>,
        conversion: Option<&AstNode<'a>>,
        parameter_a: &Parameter<'a>,
        parameter_b: &Parameter<'a>,
    ) -> AstNode<'a> {
        let geometric_product_result = result_of_trait!(geometric_product);
        let geometric_product_2_result = result_of_trait!(geometric_product_2);
        let involution_result = result_of_trait!(involution);
        let product = Box::new(Expression {
            size: 1,
            content: ExpressionContent::InvokeInstanceMethod(
                geometric_product_result.data_type.clone(),
                Box::new(Expression {
                    size: 1,
                    content: ExpressionContent::InvokeInstanceMethod(
                        parameter_a.data_type.clone(),
                        Box::new(Expression {
                            size: 1,
                            content: ExpressionContent::Variable(parameter_a.name),
                        }),
                        geometric_product_result.name,
                        vec![(
                            DataType::MultiVector(parameter_b.multi_vector_class()),
                            Expression {
                                size: 1,
                                content: ExpressionContent::Variable(parameter_b.name),
                            },
                        )],
                    ),
                }),
                geometric_product_2_result.name,
                vec![(
                    DataType::MultiVector(involution_result.multi_vector_class()),
                    Expression {
                        size: 1,
                        content: ExpressionContent::InvokeInstanceMethod(
                            parameter_a.data_type.clone(),
                            Box::new(Expression {
                                size: 1,
                                content: ExpressionContent::Variable(parameter_a.name),
                            }),
                            involution_result.name,
                            vec![],
                        ),
                    },
                )],
            ),
        });
        let conversion_result = if let Some(conversion) = conversion {
            result_of_trait!(conversion)
        } else {
            geometric_product_2_result
        };
        AstNode::TraitImplementation {
            result: Parameter {
                name,
                data_type: conversion_result.data_type.clone(),
            },
            parameters: vec![parameter_a.clone(), parameter_b.clone()],
            body: vec![AstNode::ReturnStatement {
                expression: if conversion.is_some() {
                    Box::new(Expression {
                        size: 1,
                        content: ExpressionContent::Conversion(
                            geometric_product_2_result.multi_vector_class(),
                            conversion_result.multi_vector_class(),
                            product,
                        ),
                    })
                } else {
                    product
                },
            }],
        }
    }
}
