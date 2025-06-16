use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct DeformConvNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
}

impl DeformConvNode {
    pub fn new<S: AsRef<str>>(name: S, input: TensorType, output: TensorType) -> Self {
        let field = OtherType::new(name, quote! {
            DeformConv2d<B>
        });
        Self {
            field,
            input,
            output,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for DeformConvNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // panic!("DeformConvNode is implemented only for 2d");
    
        imports.register("burn::nn::conv::DeformConv2d");
        imports.register("burn::nn::conv::Deformconv2dConfig");
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        quote! {}
    }

    fn into_node(self) -> Node<PS> {
        Node::DeformConv(self)
    }
}
