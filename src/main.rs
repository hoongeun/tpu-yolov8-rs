use opencv::{core, imgproc, prelude::*, types, videoio};
use std::time::Instant;
use tflite::{
    model::ConcatEmbeddingsOptions_FlatBuffersVTableOffset, op_resolver::BuiltInOpResovler,
    FlatBufferModel, Interpreter, InterpreterBuilder,
};

fn main() -> opencv::Result<()> {
    let model_path = "";
    let model = flatBufferModel::build_from_file("").expect("failed to load model");

    let resolver = BuiltinOpResolver::default();
    let mut builder =
        InterpreterBuilder::new(&model, &resolver).expect("failed to create interpreter builder");

    let delegate =
        tflite::Delegate::build("libedgetpu.so.1").expect("failed to load libedgetpu.so.1");

    builder.add_delegate(delegate);

    let mut interpreter = builder.build().expect("failed to create interpreter");

    interpreter
        .allocate_tensors()
        .expect("failed to allocate tensor");

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY);
    let opened = videoio::VideoCapture::is_opened(&cam);
    if !opened {
        panic!("unable to open default camera!");
    }

    cam_loop(&interpreter);

    Ok(())
}

fn cam_loop(interpreter: &Interpreter) {
    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;

        let input_tensor = frame_to_tensor(frame);
        let input_index = interpreter.inputs()[0];
        let input_tensor_info = interpreter
            .tensor_mut(input_index)
            .expect("failed to get input tensor");
        input_tensor_info
            .copy_from_buffer(input_tensor.data_bytes()?)
            .expect("failed to copy data to input tensor");

        let start = Instant::now();
        interpreter.invoke().expect("failed to invoke interpreter");

        let output_index = interpreter.outputs()[0];
        let output_tensor = interpreter
            .tensor(output_index)
            .expect("Failed to get output tensor");
        let output_data: Vec<f32> = output_tensor.data().to_vec();

        println!("Output: {:?}", output_data);

        // Display the frame
        videoio::imshow("Webcam", &frame)?;

        if videoio::wait_key(10)? == 113 {
            break;
        }
    }
}

fn frame_to_tensor(frame: &Mat) -> Option<Tensor> {
    if !frame.size()?.width == 0 {
        Some(None)
    }

    let size = core::Size();
    let mut resized_frame = Mat::default();
    imgproc::resize(
        &frame,
        &mut resized_frame,
        size,
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut input_tensor = core::Mat::default()?;
    resized_frame.convert_to(&mut input_tensor, core::CV_32F, 1.0 / 255.0, 0.0)?;
    input_tensor
}
