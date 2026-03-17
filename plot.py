from validator.graphics import plot_all


if __name__ == "__main__":
    # nc_file = r"outputs\0-ISMN.soil_moisture_with_1-SMAP_L3.soil_moisture_with_2-Smerge.soil_moisture.nc"
    nc_file = r"outputs\0-ISMN.soil_moisture_with_1-Smerge.soil_moisture_with_2-FY3B.soil_moisture_with_3-FY3C.soil_moisture_with_4-FY3D.soil_moisture.nc"
    output_dir = r"outputs\plots" + "/" + nc_file.strip(".nc").split("\\")[-1]
    plot_all(nc_file, out_dir=output_dir)
