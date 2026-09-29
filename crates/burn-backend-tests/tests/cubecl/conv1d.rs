use super::*;
use burn_tensor::{Distribution, Tolerance, module, ops::ConvOptions};

#[test]
fn conv1d_end_padding_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<3>::random([2, 4, 9], Distribution::Default, &device);
    let weight = TestTensor::<3>::random([6, 2, 4], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([6], Distribution::Default, &device);

    let input_ref = TestTensor::<3>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<3>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = ConvOptions::new_with_padding([2], [(0, 3)], [2], 2);
    let output = module::conv1d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv1d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// A dense kernel wider than a pixel, the shape `conv_im2col` exists for: a
/// causal, dilated kernel-7 convolution like a vocoder's residual units.
#[test]
fn conv1d_dense_dilated_causal_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<3>::random([1, 24, 50], Distribution::Default, &device);
    let weight = TestTensor::<3>::random([24, 24, 7], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([24], Distribution::Default, &device);

    let input_ref = TestTensor::<3>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<3>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    // Left padding only, of the kernel's whole reach: every output reads its
    // own position and six before it, three apart.
    let options = ConvOptions::new_with_padding([1], [(18, 0)], [3], 1);
    let output = module::conv1d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv1d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// The same, strided and padded both sides, over a batch: Whisper's second
/// convolution, scaled down.
#[test]
fn conv1d_dense_strided_padded_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<3>::random([3, 16, 31], Distribution::Default, &device);
    let weight = TestTensor::<3>::random([20, 16, 3], Distribution::Default, &device);

    let input_ref = TestTensor::<3>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<3>::from_data(weight.to_data(), &ref_device);

    let options = ConvOptions::new([2], [1], [1], 1);
    let output = module::conv1d(input, weight, None, options.clone());
    let output_ref = module::conv1d(input_ref, weight_ref, None, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}
