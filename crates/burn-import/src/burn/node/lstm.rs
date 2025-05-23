use super::{Node, NodeCodegen, SerializationBackend};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, Type};
use burn::{
    module::{ConstantRecord, Module, Param, ParamId},
    nn::{GateController, GateControllerRecord, LinearRecord, LstmRecord, LstmRecordItem},
    prelude::Backend,
    record::{PrecisionSettings, Record},
    tensor::{Device, Tensor, TensorData, s},
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::{Serialize, Serializer};

#[derive(Debug, Clone)]
pub struct LstmNode {
    pub field: OtherType,
    pub input: TensorType,
    pub weight: TensorData,
    pub r_weight: TensorData,
    pub bias: Option<TensorData>,
    pub sequence_lens: Option<TensorData>,
    pub initial_hidden: Option<TensorType>,
    pub initial_cell: Option<TensorType>,
    pub output: TensorType,
    pub output_hidden: Option<TensorType>,
    pub output_cell: Option<TensorType>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl LstmNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        weight: TensorData,
        r_weight: TensorData,
        bias: Option<TensorData>,
        sequence_lens: Option<TensorData>,
        initial_hidden: Option<TensorType>,
        initial_cell: Option<TensorType>,
        output: TensorType,
        output_hidden: Option<TensorType>,
        output_cell: Option<TensorType>,
        input_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            field: OtherType::new(name, quote! {
                Lstm<B>
            }),
            input,
            weight,
            r_weight,
            bias,
            sequence_lens,
            initial_hidden,
            initial_cell,
            output,
            output_hidden,
            output_cell,
            input_size,
            hidden_size,
        }
    }
}

impl LstmNode {
    fn serialize_gate_controller_record(
        input_weight: Tensor<SerializationBackend, 2>,
        input_bias: Option<Tensor<SerializationBackend, 1>>,
        hidden_weight: Tensor<SerializationBackend, 2>,
        hidden_bias: Option<Tensor<SerializationBackend, 1>>,
    ) -> GateControllerRecord<SerializationBackend> {
        GateControllerRecord::<SerializationBackend> {
            input_transform: LinearRecord::<SerializationBackend> {
                weight: Param::initialized(ParamId::new(), input_weight.transpose()),
                bias: input_bias.map(|input_bias| Param::initialized(ParamId::new(), input_bias)),
            },
            hidden_transform: LinearRecord::<SerializationBackend> {
                weight: Param::initialized(ParamId::new(), hidden_weight.transpose()),
                bias: hidden_bias
                    .map(|hidden_bias| Param::initialized(ParamId::new(), hidden_bias)),
            },
        }
    }

    fn serialize_lstm_record<PS: PrecisionSettings>(
        &self,
        device: &Device<SerializationBackend>,
        direction: usize,
    ) -> LstmRecordItem<SerializationBackend, PS> {
        // Get weight tensors for each gate
        let w_tensor: Tensor<_, 2> =
            Tensor::<_, 3, _>::from_data(self.weight.clone().convert::<PS::FloatElem>(), device)
                .slice(s![direction, ..])
                .squeeze(0);
        let w_tensors = w_tensor
            .split(self.hidden_size, 0)
            .into_iter()
            .collect::<Vec<Tensor<_, 2>>>();
        let [wi, wo, wf, wc]: [Tensor<_, 2>; 4] = w_tensors
            .try_into()
            .expect("expected w tensor to have 4 gate weights");

        // Get recurrent weight tensors for each gate
        let r_tensor: Tensor<_, 2> =
            Tensor::<_, 3, _>::from_data(self.r_weight.clone().convert::<PS::FloatElem>(), device)
                .slice(s![direction, ..])
                .squeeze(0);
        let r_tensors = r_tensor
            .split(self.hidden_size, 0)
            .into_iter()
            .collect::<Vec<Tensor<_, 2>>>();
        let [ri, ro, rf, rc]: [Tensor<_, 2>; 4] = r_tensors
            .try_into()
            .expect("expected r tensor to have 4 gate weights");

        // Get bias tensors for each gate
        let b_tensor: Option<Tensor<_, 1>> = self.bias.clone().map(|bias| {
            Tensor::<_, 2, _>::from_data(bias.convert::<PS::FloatElem>(), device)
                .slice(s![direction, ..])
                .squeeze(0)
        });
        let b_tensors = b_tensor.map(|bias_tensor| {
            bias_tensor
                .split(self.hidden_size, 0)
                .into_iter()
                .collect::<Vec<Tensor<_, 1>>>()
        });
        let [bi, bo, bf, bc, rbi, rbo, rbf, rbc]: [Option<Tensor<_, 1>>; 8] =
            if let Some(b_tensors) = b_tensors {
                let b_tensors: [Tensor<_, 1>; 8] = b_tensors
                    .try_into()
                    .expect("expected w tensor to have 8 gate weights");
                b_tensors.map(&mut |b_param| Some(b_param))
            } else {
                [const { None }; 8]
            };

        let record = LstmRecord::<SerializationBackend> {
            input_gate: Self::serialize_gate_controller_record(wi, bi, ri, rbi),
            forget_gate: Self::serialize_gate_controller_record(wf, bf, rf, rbf),
            output_gate: Self::serialize_gate_controller_record(wo, bo, ro, rbo),
            cell_gate: Self::serialize_gate_controller_record(wc, bc, rc, rbc),
            d_hidden: ConstantRecord::new(),
        };

        Record::into_item::<PS>(record)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for LstmNode {
    fn input_types(&self) -> Vec<Type> {
        let mut inputs = vec![
            Type::Tensor(self.input.clone()),
            // Type::Tensor(self.w.clone()), Type::Tensor(self.r.clone())
        ];

        // if let Some(ref b) = self.b {
        //     inputs.push(Type::Tensor(b.clone()));
        // }
        // if let Some(ref sequence_lens) = self.sequence_lens {
        //     inputs.push(Type::Tensor(sequence_lens.clone()));
        // }
        if let Some(ref initial_h) = self.initial_hidden {
            inputs.push(Type::Tensor(initial_h.clone()));
        }
        if let Some(ref initial_c) = self.initial_cell {
            inputs.push(Type::Tensor(initial_c.clone()));
        }

        inputs
    }

    fn output_types(&self) -> Vec<Type> {
        let mut outputs = vec![Type::Tensor(self.output.clone())];
        if let Some(ref output_hidden) = self.output_hidden {
            outputs.push(Type::Tensor(output_hidden.clone()));
        }
        if let Some(ref output_cell) = self.output_cell {
            outputs.push(Type::Tensor(output_cell.clone()));
        }
        outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let field = &self.field.name;

        let d_hidden = self.hidden_size;

        let initial_cell = if let Some(ref initial_cell) = self.initial_cell {
            let initial_cell = scope.tensor_use_owned(initial_cell, node_position);
            quote! { #initial_cell.slice(s![0, ..]).squeeze(0) }
        } else {
            quote! { Tensor::zeros([batch_size, #d_hidden], device) }
        };

        let initial_hidden = if let Some(ref initial_hidden) = self.initial_hidden {
            let initial_hidden = scope.tensor_use_owned(initial_hidden, node_position);
            quote! { #initial_hidden.slice(s![0, ..]).squeeze(0) }
        } else {
            quote! { Tensor::zeros([batch_size, #d_hidden], device) }
        };

        let unpack_outputs = <LstmNode as NodeCodegen<PS>>::output_types(&self)
            .iter()
            .map(|t| t.name().clone())
            .collect::<Vec<_>>();

        let mut output_values = vec![quote! { output }];
        if self.output_hidden.is_some() {
            output_values.push(quote! { hidden });
        }
        if self.output_cell.is_some() {
            output_values.push(quote! { cell });
        }

        quote! {
            let (#(#unpack_outputs),*) = {
                let [batch_size, _, _] = #input.dims();
                let initial_state = LstmState::new(#initial_cell, #initial_hidden);
                let (output, state) = self.#field.forward(#input, Some(initial_state));
                let LstmState { cell, hidden } = state;
                let output: Tensor<B, 4> = output.unsqueeze_dim(1).permute([2, 0, 1, 3]);
                (#(#output_values),*)
            };
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Lstm(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::Initializer");
        imports.register("burn::nn::GateController");
        imports.register("burn::nn::Lstm");
        imports.register("burn::nn::LstmState");
        imports.register("burn::nn::LstmConfig");
        imports.register("burn::prelude::s");
    }

    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;

        let d_input = self.input_size;
        let d_hidden = self.hidden_size;

        let tokens = quote! {
            let #name = LstmConfig::new(#d_input, #d_hidden, true)
                .with_initializer(Initializer::Zeros)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let lstm_item = self.serialize_lstm_record::<PS>(&device, 0);
        lstm_item.serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::{
        nn::{BiLstmConfig, LstmConfig},
        record::FullPrecisionSettings,
    };

    #[test]
    fn test_codegen_forward() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(LstmNode::new(
            "lstm",
            TensorType::new_float("input", 3),
            TensorData::from([2f32]),
            TensorData::from([1f32]),
            None,
            None,
            None,
            None,
            TensorType::new_float("output", 3),
            None,
            None,
            20,
            128,
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::nn::Lstm;
            use burn::nn::LstmConfig;
            use burn::nn::LstmState;

            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                lstm: Lstm<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }
            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let lstm = LstmConfig::new(20usize, 128usize).init(device);
                    Self {
                        lstm,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    input: Tensor<B, 3>,
                    initial_h: Tensor<B, 3>,
                    initial_c: Tensor<B, 3>,
                ) -> Tensor<B, 3> {
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_forward_with_inputs_outputs() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(LstmNode::new(
            "lstm",
            TensorType::new_float("input", 3),
            TensorData::from([2f32]),
            TensorData::from([1f32]),
            None,
            None,
            Some(TensorType::new_float("initial_h", 3)),
            Some(TensorType::new_float("initial_c", 3)),
            TensorType::new_float("output", 3),
            Some(TensorType::new_float("output_h", 3)),
            Some(TensorType::new_float("output_c", 3)),
            20,
            128,
        ));

        graph.register_input_output(
            vec![
                "input".to_string(),
                "initial_h".to_string(),
                "initial_c".to_string(),
            ],
            vec![
                "output".to_string(),
                "output_h".to_string(),
                "output_c".to_string(),
            ],
        );

        let expected = quote! {
            use burn::nn::Lstm;
            use burn::nn::LstmConfig;
            use burn::nn::LstmState;

            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                lstm: Lstm<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }
            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let lstm = LstmConfig::new(20usize, 128usize).init(device);
                    Self {
                        lstm,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    input: Tensor<B, 3>,
                    initial_h: Tensor<B, 3>,
                    initial_c: Tensor<B, 3>,
                ) -> Tensor<B, 3> {
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
