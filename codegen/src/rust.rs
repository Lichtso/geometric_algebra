use crate::{
    ast::{AstNode, DataType, Expression, ExpressionContent},
    emit::{camel_to_snake_case, emit_indentation},
};

fn emit_data_type<W: std::io::Write>(collector: &mut W, data_type: &DataType) -> std::io::Result<()> {
    match data_type {
        DataType::Integer => collector.write_all(b"isize"),
        DataType::SimdVector(size) => collector.write_fmt(format_args!("Simd32x{}", size)),
        DataType::MultiVector(class) => collector.write_fmt(format_args!("{}", class.class_name)),
    }
}

fn emit_expression<W: std::io::Write>(collector: &mut W, expression: &Expression) -> std::io::Result<()> {
    match &expression.content {
        ExpressionContent::None => unreachable!(),
        ExpressionContent::Variable(name) => {
            collector.write_all(name.bytes().collect::<Vec<_>>().as_slice())?;
        }
        ExpressionContent::InvokeClassMethod(_, method_name, arguments) | ExpressionContent::InvokeInstanceMethod(_, _, method_name, arguments) => {
            match &expression.content {
                ExpressionContent::InvokeInstanceMethod(_result_class, inner_expression, _, _) => {
                    emit_expression(collector, &inner_expression)?;
                    collector.write_all(b".")?;
                }
                ExpressionContent::InvokeClassMethod(class, _, _) => {
                    if *method_name == "Constructor" {
                        collector.write_fmt(format_args!("{} {{ ", class.class_name))?;
                    } else {
                        collector.write_fmt(format_args!("{}::", class.class_name))?;
                    }
                }
                _ => unreachable!(),
            }
            if *method_name != "Constructor" {
                camel_to_snake_case(collector, method_name)?;
                collector.write_all(b"(")?;
            }
            for (i, (_argument_class, argument)) in arguments.iter().enumerate() {
                if i > 0 {
                    collector.write_all(b", ")?;
                }
                if *method_name == "Constructor" {
                    collector.write_fmt(format_args!("g{}: ", i))?;
                }
                emit_expression(collector, &argument)?;
            }
            if *method_name == "Constructor" {
                collector.write_all(b" }")?;
            } else {
                collector.write_all(b")")?;
            }
        }
        ExpressionContent::Conversion(_source_class, _destination_class, inner_expression) => {
            emit_expression(collector, &inner_expression)?;
            collector.write_all(b".into()")?;
        }
        ExpressionContent::Select(condition_expression, then_expression, else_expression) => {
            collector.write_all(b"if ")?;
            emit_expression(collector, &condition_expression)?;
            collector.write_all(b" { ")?;
            emit_expression(collector, &then_expression)?;
            collector.write_all(b" } else { ")?;
            emit_expression(collector, &else_expression)?;
            collector.write_all(b" }")?;
        }
        ExpressionContent::Access(inner_expression, array_index) => {
            emit_expression(collector, &inner_expression)?;
            collector.write_fmt(format_args!(".g{}", array_index))?;
        }
        ExpressionContent::Swizzle(inner_expression, indices) => {
            if expression.size == 1 {
                emit_expression(collector, &inner_expression)?;
                if inner_expression.size > 1 {
                    collector.write_fmt(format_args!(".get_f({})", indices[0]))?;
                }
            } else {
                collector.write_all(b"swizzle!(")?;
                emit_expression(collector, &inner_expression)?;
                collector.write_all(b", ")?;
                for (i, component_index) in indices.iter().enumerate() {
                    if i > 0 {
                        collector.write_all(b", ")?;
                    }
                    collector.write_fmt(format_args!("{}", *component_index))?;
                }
                collector.write_all(b")")?;
            }
        }
        ExpressionContent::Gather(inner_expression, indices) => {
            if expression.size > 1 {
                collector.write_fmt(format_args!("Simd32x{}::from(", expression.size))?;
            }
            if indices.len() > 1 {
                collector.write_all(b"[")?;
            }
            for (i, (array_index, component_index)) in indices.iter().enumerate() {
                if i > 0 {
                    collector.write_all(b", ")?;
                }
                emit_expression(collector, &inner_expression)?;
                collector.write_fmt(format_args!(".g{}", array_index))?;
                if inner_expression.size > 1 {
                    collector.write_fmt(format_args!(".get_f({})", *component_index))?;
                }
            }
            if indices.len() > 1 {
                collector.write_all(b"]")?;
            }
            if expression.size > 1 {
                collector.write_all(b")")?;
            }
        }
        ExpressionContent::Constant(data_type, values) => match data_type {
            DataType::Integer => collector.write_fmt(format_args!("{}", values[0] as f32))?,
            DataType::SimdVector(_size) => {
                if expression.size == 1 {
                    collector.write_fmt(format_args!("{:.1}", values[0] as f32))?;
                } else {
                    collector.write_fmt(format_args!("Simd32x{}::from(", expression.size))?;
                    if values.len() > 1 {
                        collector.write_all(b"[")?;
                    }
                    for (i, value) in values.iter().enumerate() {
                        if i > 0 {
                            collector.write_all(b", ")?;
                        }
                        collector.write_fmt(format_args!("{:.1}", *value as f32))?;
                    }
                    if values.len() > 1 {
                        collector.write_all(b"]")?;
                    }
                    collector.write_all(b")")?;
                }
            }
            _ => unreachable!(),
        },
        ExpressionContent::SquareRoot(inner_expression) => {
            emit_expression(collector, &inner_expression)?;
            collector.write_all(b".sqrt()")?;
        }
        ExpressionContent::Add(lhs, rhs)
        | ExpressionContent::Subtract(lhs, rhs)
        | ExpressionContent::Multiply(lhs, rhs)
        | ExpressionContent::Divide(lhs, rhs)
        | ExpressionContent::LessThan(lhs, rhs)
        | ExpressionContent::Equal(lhs, rhs)
        | ExpressionContent::LogicAnd(lhs, rhs)
        | ExpressionContent::BitShiftRight(lhs, rhs) => {
            emit_expression(collector, &lhs)?;
            collector.write_all(match expression.content {
                ExpressionContent::Add(_, _) => b" + ",
                ExpressionContent::Subtract(_, _) => b" - ",
                ExpressionContent::Multiply(_, _) => b" * ",
                ExpressionContent::Divide(_, _) => b" / ",
                ExpressionContent::LessThan(_, _) => b" < ",
                ExpressionContent::Equal(_, _) => b" == ",
                ExpressionContent::LogicAnd(_, _) => b" & ",
                ExpressionContent::BitShiftRight(_, _) => b" >> ",
                _ => unreachable!(),
            })?;
            emit_expression(collector, &rhs)?;
        }
    }
    Ok(())
}

pub fn emit_code<W: std::io::Write>(collector: &mut W, ast_node: &AstNode, indentation: usize) -> std::io::Result<()> {
    match &ast_node {
        AstNode::None => {}
        AstNode::Preamble => {
            collector.write_all(b"#![allow(clippy::assign_op_pattern)]\n")?;
            collector.write_all(b"use crate::*;\nuse std::ops::{Add, Neg, Sub, Mul, Div};\n\n")?;
        }
        AstNode::ClassDefinition { class } => {
            collector.write_fmt(format_args!("#[derive(Clone, Copy)]\npub struct {} {{\n", class.class_name))?;
            for (i, group) in class.grouped_basis.iter().enumerate() {
                emit_indentation(collector, indentation + 1)?;
                collector.write_all(b"/// ")?;
                for (i, element) in group.iter().enumerate() {
                    if i > 0 {
                        collector.write_all(b", ")?;
                    }
                    collector.write_fmt(format_args!("{}", element))?;
                }
                collector.write_all(b"\n")?;
                emit_indentation(collector, indentation + 1)?;
                collector.write_fmt(format_args!("pub g{}: ", i))?;
                if group.len() == 1 {
                    collector.write_all(b"f32,\n")?;
                } else {
                    collector.write_fmt(format_args!("Simd32x{},\n", group.len()))?;
                }
            }
            collector.write_all(b"}\n\n")?;
        }
        AstNode::ReturnStatement { expression } => {
            collector.write_all(b"return ")?;
            emit_expression(collector, expression)?;
            collector.write_all(b";\n")?;
        }
        AstNode::VariableAssignment { name, data_type, expression } => {
            if let Some(data_type) = data_type {
                collector.write_fmt(format_args!("let mut {}", name))?;
                collector.write_all(b": ")?;
                emit_data_type(collector, data_type)?;
            } else {
                collector.write_fmt(format_args!("{}", name))?;
            }
            collector.write_all(b" = ")?;
            emit_expression(collector, expression)?;
            collector.write_all(b";\n")?;
        }
        AstNode::IfThenBlock { condition, body } | AstNode::WhileLoopBlock { condition, body } => {
            collector.write_all(match &ast_node {
                AstNode::IfThenBlock { .. } => b"if ",
                AstNode::WhileLoopBlock { .. } => b"while ",
                _ => unreachable!(),
            })?;
            emit_expression(collector, condition)?;
            collector.write_all(b" {\n")?;
            for statement in body.iter() {
                emit_indentation(collector, indentation + 1)?;
                emit_code(collector, statement, indentation + 1)?;
            }
            emit_indentation(collector, indentation)?;
            collector.write_all(b"}\n")?;
        }
        AstNode::TraitImplementation { result, parameters, body } => {
            match parameters.len() {
                0 => collector.write_fmt(format_args!("impl {} for {}", result.name, result.multi_vector_class().class_name))?,
                1 if result.name == "Into" => collector.write_fmt(format_args!(
                    "impl {}<{}> for {}",
                    result.name,
                    result.multi_vector_class().class_name,
                    parameters[0].multi_vector_class().class_name,
                ))?,
                1 => collector.write_fmt(format_args!("impl {} for {}", result.name, parameters[0].multi_vector_class().class_name))?,
                2 if result.name == "Powi" => {
                    collector.write_fmt(format_args!("impl {} for {}", result.name, parameters[0].multi_vector_class().class_name))?
                }
                2 => collector.write_fmt(format_args!(
                    "impl {}<{}> for {}",
                    result.name,
                    parameters[1].multi_vector_class().class_name,
                    parameters[0].multi_vector_class().class_name,
                ))?,
                _ => unreachable!(),
            }
            collector.write_all(b" {\n")?;
            if !parameters.is_empty() && result.name != "Into" {
                emit_indentation(collector, indentation + 1)?;
                collector.write_fmt(format_args!("type Output = {};\n\n", result.multi_vector_class().class_name))?;
            }
            emit_indentation(collector, indentation + 1)?;
            collector.write_all(b"fn ")?;
            camel_to_snake_case(collector, result.name)?;
            match parameters.len() {
                0 => collector.write_all(b"() -> Self")?,
                1 => {
                    collector.write_fmt(format_args!("({}) -> ", parameters[0].name))?;
                    emit_data_type(collector, &result.data_type)?;
                }
                2 => {
                    collector.write_fmt(format_args!("({}, {}: ", parameters[0].name, parameters[1].name))?;
                    emit_data_type(collector, &parameters[1].data_type)?;
                    collector.write_all(b") -> ")?;
                    emit_data_type(collector, &result.data_type)?;
                }
                _ => unreachable!(),
            }
            collector.write_all(b" {\n")?;
            for (i, statement) in body.iter().enumerate() {
                emit_indentation(collector, indentation + 2)?;
                if i + 1 == body.len() {
                    if let AstNode::ReturnStatement { expression } = statement {
                        emit_expression(collector, expression)?;
                        collector.write_all(b"\n")?;
                        break;
                    }
                }
                emit_code(collector, statement, indentation + 2)?;
            }
            emit_indentation(collector, indentation + 1)?;
            collector.write_all(b"}\n}\n\n")?;
        }
    }
    Ok(())
}
