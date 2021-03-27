pub struct GeometricAlgebra<'a> {
    pub generator_squares: &'a [isize],
}

impl<'a> GeometricAlgebra<'a> {
    pub fn basis_size(&self) -> usize {
        1 << self.generator_squares.len()
    }

    pub fn basis(&self) -> impl Iterator<Item = BasisElement> + '_ {
        (0..self.basis_size() as BasisElementIndex).map(|index| BasisElement { index })
    }

    pub fn sorted_basis(&self) -> Vec<BasisElement> {
        let mut basis_elements = self.basis().collect::<Vec<BasisElement>>();
        basis_elements.sort();
        basis_elements
    }
}

type BasisElementIndex = u16;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BasisElement {
    pub index: BasisElementIndex,
}

impl BasisElement {
    pub fn new(name: &str) -> Self {
        Self {
            index: if name == "1" {
                0
            } else {
                let mut generator_indices = name.chars();
                assert_eq!(generator_indices.next().unwrap(), 'e');
                generator_indices.fold(0, |index, generator_index| index | (1 << (generator_index.to_digit(16).unwrap())))
            },
        }
    }

    pub fn grade(&self) -> usize {
        self.index.count_ones() as usize
    }

    pub fn component_bits(&self) -> impl Iterator<Item = usize> + '_ {
        (0..std::mem::size_of::<BasisElementIndex>() * 8).filter(move |index| (self.index >> index) & 1 != 0)
    }

    pub fn dual(&self, algebra: &GeometricAlgebra) -> Self {
        Self {
            index: algebra.basis_size() as BasisElementIndex - 1 - self.index,
        }
    }
}

impl std::fmt::Display for BasisElement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.index == 0 {
            formatter.pad("1")
        } else {
            let string = format!("e{}", self.component_bits().map(|index| format!("{:X}", index)).collect::<String>());
            formatter.pad(string.as_str())
        }
    }
}

impl std::cmp::Ord for BasisElement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let grades_order = self.grade().cmp(&other.grade());
        if grades_order != std::cmp::Ordering::Equal {
            return grades_order;
        }
        let a_without_b = self.index & (!other.index);
        let b_without_a = other.index & (!self.index);
        if a_without_b.trailing_zeros() < b_without_a.trailing_zeros() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }
}

impl std::cmp::PartialOrd for BasisElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, PartialEq)]
pub struct ScaledElement {
    pub scalar: isize,
    pub unit: BasisElement,
}

impl ScaledElement {
    pub fn from(element: &BasisElement) -> Self {
        Self {
            scalar: 1,
            unit: element.clone(),
        }
    }

    pub fn product(a: &Self, b: &Self, algebra: &GeometricAlgebra) -> Self {
        let commutations = a
            .unit
            .component_bits()
            .fold((0, a.unit.index, b.unit.index), |(commutations, a, b), index| {
                let hurdles_a = a & (BasisElementIndex::MAX << (index + 1));
                let hurdles_b = b & ((1 << index) - 1);
                (
                    commutations
                        + BasisElement {
                            index: hurdles_a | hurdles_b,
                        }
                        .grade(),
                    a & !(1 << index),
                    b ^ (1 << index),
                )
            });
        Self {
            scalar: BasisElement {
                index: a.unit.index & b.unit.index,
            }
            .component_bits()
            .map(|i| algebra.generator_squares[i])
            .fold(a.scalar * b.scalar * if commutations.0 % 2 == 0 { 1 } else { -1 }, |a, b| a * b),
            unit: BasisElement {
                index: a.unit.index ^ b.unit.index,
            },
        }
    }
}

impl std::fmt::Display for ScaledElement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let string = self.unit.to_string();
        formatter.pad_integral(self.scalar >= 0, "", if self.scalar == 0 { "0" } else { string.as_str() })
    }
}

#[derive(Clone)]
pub struct Involution {
    pub terms: Vec<ScaledElement>,
}

impl Involution {
    pub fn identity(algebra: &GeometricAlgebra) -> Self {
        Self {
            terms: algebra.basis().map(|element| ScaledElement { scalar: 1, unit: element }).collect(),
        }
    }

    pub fn negated<F>(&self, grade_negation: F) -> Self
    where
        F: Fn(usize) -> bool,
    {
        Self {
            terms: self
                .terms
                .iter()
                .map(|element| ScaledElement {
                    scalar: if grade_negation(element.unit.grade()) { -1 } else { 1 },
                    unit: element.unit.clone(),
                })
                .collect(),
        }
    }

    pub fn dual(&self, algebra: &GeometricAlgebra) -> Self {
        Self {
            terms: self
                .terms
                .iter()
                .map(|term| ScaledElement {
                    scalar: term.scalar,
                    unit: term.unit.dual(algebra),
                })
                .collect(),
        }
    }

    pub fn involutions(algebra: &GeometricAlgebra) -> Vec<(&'static str, Self)> {
        let involution = Self::identity(algebra);
        vec![
            ("Neg", involution.negated(|_grade| true)),
            ("Automorph", involution.negated(|grade| grade % 2 == 1)),
            ("Transpose", involution.negated(|grade| grade % 4 >= 2)),
            ("Conjugate", involution.negated(|grade| (grade + 3) % 4 < 2)),
            ("Dual", involution.dual(algebra)),
        ]
    }
}

#[derive(Clone, PartialEq)]
pub struct ProductTerm {
    pub product: ScaledElement,
    pub factor_a: BasisElement,
    pub factor_b: BasisElement,
}

#[derive(Clone)]
pub struct Product {
    pub terms: Vec<ProductTerm>,
}

impl Product {
    pub fn product(a: &[ScaledElement], b: &[ScaledElement], algebra: &GeometricAlgebra) -> Self {
        Self {
            terms: a
                .iter()
                .map(|a| {
                    b.iter().map(move |b| ProductTerm {
                        product: ScaledElement::product(&a, &b, algebra),
                        factor_a: a.unit.clone(),
                        factor_b: b.unit.clone(),
                    })
                })
                .flatten()
                .filter(|term| term.product.scalar != 0)
                .collect(),
        }
    }

    pub fn projected<F>(&self, grade_projection: F) -> Self
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        Self {
            terms: self
                .terms
                .iter()
                .filter(|term| grade_projection(term.factor_a.grade(), term.factor_b.grade(), term.product.unit.grade()))
                .cloned()
                .collect(),
        }
    }

    pub fn dual(&self, algebra: &GeometricAlgebra) -> Self {
        Self {
            terms: self
                .terms
                .iter()
                .map(|term| ProductTerm {
                    product: ScaledElement {
                        scalar: term.product.scalar,
                        unit: term.product.unit.dual(algebra),
                    },
                    factor_a: term.factor_a.dual(algebra),
                    factor_b: term.factor_b.dual(algebra),
                })
                .collect(),
        }
    }

    pub fn products(algebra: &GeometricAlgebra) -> Vec<(&'static str, Self)> {
        let basis = algebra.basis().map(|element| ScaledElement::from(&element)).collect::<Vec<_>>();
        let product = Self::product(&basis, &basis, algebra);
        vec![
            ("GeometricProduct", product.clone()),
            ("RegressiveProduct", product.projected(|r, s, t| t == r + s).dual(algebra)),
            ("OuterProduct", product.projected(|r, s, t| t == r + s)),
            ("InnerProduct", product.projected(|r, s, t| t == (r as isize - s as isize).abs() as usize)),
            ("LeftContraction", product.projected(|r, s, t| t as isize == s as isize - r as isize)),
            ("RightContraction", product.projected(|r, s, t| t as isize == r as isize - s as isize)),
            ("ScalarProduct", product.projected(|_r, _s, t| t == 0)),
        ]
    }
}

#[derive(Default)]
pub struct MultiVectorClassRegistry {
    pub classes: Vec<MultiVectorClass>,
    index_by_signature: std::collections::HashMap<Vec<BasisElement>, usize>,
}

impl MultiVectorClassRegistry {
    pub fn register(&mut self, class: MultiVectorClass) {
        self.index_by_signature.insert(class.signature(), self.classes.len());
        self.classes.push(class);
    }

    pub fn get(&self, signature: &[BasisElement]) -> Option<&MultiVectorClass> {
        self.index_by_signature.get(signature).map(|index| &self.classes[*index])
    }
}

#[derive(PartialEq, Eq)]
pub struct MultiVectorClass {
    pub class_name: String,
    pub grouped_basis: Vec<Vec<BasisElement>>,
}
