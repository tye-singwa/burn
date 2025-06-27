use super::{Node, NodeCodegen};
use crate::burn::{
    BurnImports, OtherType, Scope, TensorType, ToTokens, Type, node::SerializationBackend,
};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::{LinearRecord, attention::MultiHeadAttentionRecord, conv::ConvTranspose3dRecord},
    prelude::Backend,
    record::{PrecisionSettings, Record},
    tensor::{Device, Tensor, TensorData},
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct AttentionNode {
    pub field: OtherType,
    pub q: TensorType,
    pub k: TensorType,
    pub v: TensorType,
    pub output: TensorType,
}

impl AttentionNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        q: TensorType,
        k: TensorType,
        v: TensorType,
        output: TensorType,
    ) -> Self {
        let tokens = quote! { MultiHeadAttention<B> };

        Self {
            field: OtherType::new(name, tokens),
            q,
            k,
            v,
            output,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for AttentionNode {
    fn input_types(&self) -> Vec<Type> {
        vec![
            Type::Tensor(self.q.clone()),
            Type::Tensor(self.k.clone()),
            Type::Tensor(self.v.clone()),
        ]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::attention::MultiHeadAttentionConfig");
        imports.register("burn::nn::attention::MultiHeadAttention");
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;
        let d_model = 0;
        let n_heads = 0;

        let tokens = quote! {
            let #name = MultiHeadAttentionConfig::new(#d_model, #n_heads)
            .with_initializer(Initializer::Zeros)
            .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = <SerializationBackend as Backend>::Device::default();
        let record = MultiHeadAttentionRecord::<SerializationBackend> {
            query: todo!(),
            key: todo!(),
            value: todo!(),
            output: todo!(),
            dropout: todo!(),
            activation: todo!(),
            d_model: todo!(),
            n_heads: todo!(),
            d_k: todo!(),
            min_float: todo!(),
            quiet_softmax: todo!(),
        };

        let item = Record::into_item::<PS>(record);
        item.serialize(serializer)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        quote! {}
    }

    fn into_node(self) -> Node<PS> {
        Node::Attention(self)
    }
}
