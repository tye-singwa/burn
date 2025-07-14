use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

#[derive(Debug, Clone, new)]
pub struct EyeLikeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub offset: i64,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for EyeLikeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::prelude::Bool");
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let offset = self.offset;

        let mut output_expr = quote! {
            let dims: [usize; 2] = #input.shape().dims();
            let mask = Tensor::<B, 2, Bool>::diag_mask(dims, #offset, &self.device).bool_not();
            #input.zeros_like().mask_fill(mask, 1)
        };

        if self.input.kind != self.output.kind {
            output_expr = match self.output.kind {
                TensorKind::Int => quote! { #output_expr.int() },
                TensorKind::Float => quote! { #output_expr.float() },
                TensorKind::Bool => quote! { #output_expr.bool() },
            }
        }

        quote! {
            let #output = { #output_expr };
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::EyeLike(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(EyeLikeNode::new(
            TensorType::new_float("input1", 2),
            TensorType::new_float("output1", 2),
            4,
        ));

        graph.register_input_output(vec!["input1".to_string()], vec!["output1".to_string()]);

        let expected = quote! {
            use burn::prelude::Bool;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input1: Tensor<B, 2>) -> Tensor<B, 2> {
                    let output1 = {
                        let input_dims: [usize; _] = input1.shape().dims();
                        let mask =
                            Tensor::<B, 2, Bool>::diag_mask([input_dims[0], input_dims[1]], 4i64, &self.device);
                        input1.zeros_like().mask_fill(mask, 0)
                    };

                    output1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
