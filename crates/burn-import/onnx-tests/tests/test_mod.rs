#![no_std]

extern crate alloc;

// Import individual node modules
pub mod add;
pub mod and;
pub mod argmax;
pub mod avg_pool;
pub mod batch_norm;
pub mod cast;
pub mod clip;
pub mod concat;
pub mod constant;
pub mod constant_of_shape;
pub mod conv;
pub mod conv_transpose;
pub mod cos;
pub mod cosh;
pub mod div;
pub mod dropout;
pub mod equal;
pub mod erf;
pub mod exp;
pub mod expand;
pub mod flatten;
pub mod floor;
pub mod gather;
pub mod gelu;
pub mod gemm;
pub mod global_avr_pool;
pub mod graph_multiple_output_tracking;
pub mod greater;
pub mod greater_or_equal;
pub mod hard_sigmoid;
pub mod instance_norm;
pub mod layer_norm;
pub mod leaky_relu;
pub mod less;
pub mod less_or_equal;
pub mod linear;
pub mod log;
pub mod log_softmax;
pub mod mask_where;
pub mod matmul;
pub mod max;
pub mod maxpool;
pub mod mean;
pub mod min;
pub mod mul;
pub mod neg;
pub mod non_zero;
pub mod not;
pub mod one_hot;
pub mod or;
pub mod pad;
pub mod pow;
pub mod prelu;
pub mod random_normal;
pub mod random_normal_like;
pub mod random_uniform;
pub mod random_uniform_like;
pub mod range;
pub mod recip;
pub mod reduce_max;
pub mod reduce_mean;
pub mod reduce_min;
pub mod reduce_prod;
pub mod reduce_sum;
pub mod relu;
pub mod reshape;
pub mod resize;
pub mod shape;
pub mod sigmoid;
pub mod sign;
pub mod sin;
pub mod sinh;
pub mod slice;
pub mod softmax;
pub mod split;
pub mod sqrt;
pub mod squeeze;
pub mod sub;
pub mod sum;
pub mod tan;
pub mod tanh;
pub mod tile;
pub mod topk;
pub mod transpose;
pub mod trilu;
pub mod unsqueeze;
pub mod xor;

/// Include specified models in the `model` directory in the target directory.
#[macro_export]
macro_rules! include_models {
    ($($model:ident),*) => {
        $(
            // Allow type complexity for generated code
            #[allow(clippy::type_complexity)]
            pub mod $model {
                include!(concat!(env!("OUT_DIR"), concat!("/model/", stringify!($model), ".rs")));
            }
        )*
    };
}
