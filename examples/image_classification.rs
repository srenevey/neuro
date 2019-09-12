use std::path::Path;
use arrayfire::*;

use neuro::data::DataSetError;
use neuro::data::image_data::ImageDataSet;


fn main() -> Result<(), DataSetError> {

    let path = Path::new("datasets/MNIST");
    let data = ImageDataSet::from_path(&path, 28)?;
    data.print_stats();

    Ok(())
}
