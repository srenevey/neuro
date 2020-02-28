use hdf5_sys::h5::{H5_INDEX_CRT_ORDER, H5_ITER_INC};
use hdf5_sys::h5p::{H5P_DEFAULT, H5P_CRT_ORDER_INDEXED, H5P_CRT_ORDER_TRACKED, H5P_CLS_LINK_CREATE, H5P_CLS_GROUP_CREATE};
use std::ffi::{CStr, CString};

use crate::tensor::*;

/// Creates an H5 group with creation order tracked and indexed.
///
/// # Arguments
///
/// * `file` - The file in which  the group is created.
/// * `group_name` - The name of the group.
pub(crate) fn create_group(file: &hdf5::File, group_name: &str) -> hdf5::Group {
    let name = CString::new(group_name).unwrap();
    unsafe {
        let lcpl = hdf5_sys::h5p::H5Pcreate(*H5P_CLS_LINK_CREATE);
        let gcpl = hdf5_sys::h5p::H5Pcreate(*H5P_CLS_GROUP_CREATE);
        hdf5_sys::h5p::H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
        hdf5_sys::h5g::H5Gcreate2(file.id(), name.as_ptr(), lcpl, gcpl, H5P_DEFAULT);
    }
    file.group(group_name).expect("Could not create the group.")

}

/// Lists the subgroups contained in a group.
///
/// # Return value
///
/// Vector containing the names of the subgroups, listed by creation order.
pub(crate) fn list_subgroups(group: &hdf5::Group) -> Vec<String> {
    extern "C" fn members_callback(
        _id: hdf5_sys::h5i::hid_t, name: *const std::os::raw::c_char, _info: *const hdf5_sys::h5l::H5L_info_t, op_data: *mut std::os::raw::c_void,
    ) -> hdf5_sys::h5::herr_t {
        let other_data: &mut Vec<String> = unsafe { &mut *(op_data as *mut Vec<String>) };
        unsafe {
            let name_str = CStr::from_ptr(name).to_str().unwrap().to_owned();
            other_data.push(name_str);
        }
        0
    }
    let callback_fn: hdf5_sys::h5l::H5L_iterate_t = Some(members_callback);
    let mut result: Vec<String> = Vec::new();
    let other_data: *mut std::os::raw::c_void = &mut result as *mut _ as *mut std::os::raw::c_void;
    let iteration_position: *mut hdf5_sys::h5::hsize_t = &mut { 0 as u64 };
    unsafe {
        hdf5_sys::h5l::H5Literate(group.id(), H5_INDEX_CRT_ORDER, H5_ITER_INC, iteration_position, callback_fn, other_data);
    }
    result
}

pub(crate) fn write_scalar<T: hdf5::H5Type>(dataset: &hdf5::Dataset, value: &T) {
    let mem_dtype = hdf5::Datatype::from_type::<T>().expect("Could not determine the type of the scalar.");
    let tmp = value as *const _;
    unsafe { hdf5_sys::h5d::H5Dwrite(dataset.id(), mem_dtype.id(), hdf5_sys::h5s::H5S_ALL, hdf5_sys::h5s::H5S_ALL, hdf5_sys::h5p::H5P_DEFAULT, tmp as *const _); }
}

pub(crate) fn read_scalar<T: hdf5::H5Type>(dataset: &hdf5::Dataset) -> T {
    let mem_dtype = hdf5::Datatype::from_type::<T>().expect("Could not determine the type of the scalar.");
    let mut buffer = std::mem::MaybeUninit::<T>::uninit();
    unsafe {
        hdf5_sys::h5d::H5Dread(dataset.id(), mem_dtype.id(), hdf5_sys::h5s::H5S_ALL, hdf5_sys::h5s::H5S_ALL, hdf5_sys::h5p::H5P_DEFAULT, buffer.as_mut_ptr() as *mut _);
        buffer.assume_init()
    }
}

/// Saves a slice of tensors in an HDF5 group.
///
/// # Arguments
///
/// * `group` - The group where the vector is saved.
/// * `slice` - A slice containing the tensors.
/// * `name` - The name of the dataset where the vector is saved.
pub(crate) fn save_vec_tensor(group: &hdf5::Group,
                              slice: &[Tensor],
                              name: &str
) -> hdf5::Result<()> {
    let values: Vec<H5Tensor> = slice.iter().map(H5Tensor::from).collect();
    let ds = group.new_dataset::<H5Tensor>().create(name, slice.len())?;
    ds.write(values.as_slice())?;
    Ok(())
}