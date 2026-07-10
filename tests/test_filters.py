import numpy as np

from validator.filters import (
    check_normalized_bits_array,
    get_used_variables,
    smos_exclude_bitmask,
)
from validator.models import DataFilter, DataVariable


class TestSmosExcludeBitmask:
    def test_excludes_masked_bits(self):
        data = np.array([0b0001, 0b0010, 0b0011, 0b0100])
        result = smos_exclude_bitmask(data, 0b0001)
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_all_pass_with_no_match(self):
        data = np.array([0b0010, 0b0100, 0b1000])
        result = smos_exclude_bitmask(data, 0b0001)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)


class TestCheckNormalizedBitsArray:
    def test_single_bit_active(self):
        numbers = np.array([0b001, 0b010, 0b100, 0b011])
        result = check_normalized_bits_array(numbers, [[0]])
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_either_bit_active(self):
        numbers = np.array([0b001, 0b010, 0b100])
        result = check_normalized_bits_array(numbers, [[0], [1]])
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_both_bits_active(self):
        numbers = np.array([0b001, 0b011, 0b110])
        result = check_normalized_bits_array(numbers, [[0, 1]])
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result, expected)


class TestGetUsedVariables:
    def make_var(self, name="soil_moisture"):
        return DataVariable(
            id=1,
            short_name=name,
            pretty_name=name,
            help_text="",
        )

    def make_filter(self, name):
        return DataFilter(id=1, name=name, description=name, help_text="")

    def test_default(self):
        var = self.make_var()
        result = get_used_variables([], None, var)
        assert result == ["soil_moisture"]

    def test_ismn_good(self):
        var = self.make_var()
        f = self.make_filter("FIL_ISMN_GOOD")
        result = get_used_variables([f], None, var)
        assert "soil_moisture_flag" in result

    def test_smos_qual_recommended(self):
        var = self.make_var()
        f = self.make_filter("FIL_SMOS_QUAL_RECOMMENDED")
        result = get_used_variables([f], None, var)
        assert "Quality_Flag" in result

    def test_unknown_filter_ignored(self):
        var = self.make_var()
        f = self.make_filter("FIL_NONEXISTENT")
        result = get_used_variables([f], None, var)
        assert result == ["soil_moisture"]

    def test_multiple_filters(self):
        var = self.make_var()
        result = get_used_variables(
            [self.make_filter("FIL_ISMN_GOOD"), self.make_filter("FIL_SMOS_UNFROZEN")],
            None,
            var,
        )
        assert "soil_moisture_flag" in result
        assert "Scene_Flags" in result

    def test_ascat_metop_a(self):
        var = self.make_var()
        f = self.make_filter("FIL_ASCAT_METOP_A")
        result = get_used_variables([f], None, var)
        assert "sat_id" in result

    def test_ccigf_gapmask(self):
        var = self.make_var()
        f = self.make_filter("FIL_CCIGF_GAPMASK")
        result = get_used_variables([f], None, var)
        assert "gapmask" in result

    def test_ccigf_frozenmask(self):
        var = self.make_var()
        f = self.make_filter("FIL_CCIGF_FROZENMASK")
        result = get_used_variables([f], None, var)
        assert "frozenmask" in result

    def test_smap_orbit_filters(self):
        var = self.make_var()
        f1 = self.make_filter("FIL_SMAP_L3_V9_ORBIT_ASC")
        f2 = self.make_filter("FIL_SMAP_L3_V9_ORBIT_DSC")
        result = get_used_variables([f1, f2], None, var)
        assert "Overpass" in result

    def test_smosl3_science_flags(self):
        var = self.make_var()
        for filter_name in [
            "FIL_SMOSL3_STRONG_TOPO_MANDATORY",
            "FIL_SMOSL3_MODERATE_TOPO",
            "FIL_SMOSL3_ICE_MANDATORY",
            "FIL_SMOSL3_FROZEN",
            "FIL_SMOSL3_URBAN_LOW",
            "FIL_SMOSL3_URBAN_HIGH",
            "FIL_SMOSL3_WATER",
            "FIL_SMOSL3_EXTERNAL",
            "FIL_SMOSL3_TAU_FO",
        ]:
            result = get_used_variables([self.make_filter(filter_name)], None, var)
            assert "Science_Flags" in result, f"{filter_name} should add Science_Flags"
