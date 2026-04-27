import logging
import sys
from pathlib import Path

import polars as pl
from da_hf5_utils.db2 import get_engine

from da_hf5_dz.config import OLD_DZ_RULE, AggregationIdentifiers, filepath_shared_folder
from da_hf5_dz.helpers import collect_dz_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)


GREEDY_RULES = [
    ["eq_TAB_BETR_POSITION", "overlap_0.5_clean_DIAG_ZIFFER", "eq_end_date"],
    ["eq_TAB_BETR_POSITION", "overlap_0.5_clean_DIAG_ZIFFER", "eq_start_date"],
    ["eq_TAB_BETR_POSITION", "overlap_0_clean_DIAG_ZIFFER", "eq_start_date"],
    ["overlap_0.5_TAB_DAT_LSTG_VON", "eq_clean_BEL_BETR_RECHNUNG", "eq_start_date"],
    [
        "eq_BE_NAME",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
    ],
    [
        "eq_BE_NAME",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_BETR_RECHNUNG",
        "overlap_0_clean_TAB_GEBUEHZIFFER",
    ],
    [
        "eq_BE_NAME",
        "eq_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_start_date",
    ],
    [
        "eq_BE_STRASSE_NR",
        "eq_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_start_date",
    ],
    [
        "eq_TAB_BETR_POSITION",
        "overlap_0.5_TAB_BETR_POSITION",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "eq_end_date",
    ],
    [
        "eq_TAB_BETR_POSITION",
        "overlap_0_TAB_DAT_LSTG_VON",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "dist_1_end_date",
    ],
    [
        "eq_TAB_BETR_POSITION",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "eq_clean_BEL_DAT_VERORDNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    ],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_start_date",
    ],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_BETR_RECHNUNG",
        "dist_1_start_date",
    ],
    [
        "overlap_0_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_DIAG_ZIFFER",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    ],
    [
        "overlap_0_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_end_date",
        "dist_1_start_date",
    ],
    [
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "eq_end_date",
        "dist_1_start_date",
    ],
    [
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "dist_1_end_date",
        "eq_start_date",
    ],
    [
        "eq_clean_BEL_BETR_RECHNUNG",
        "overlap_0_clean_TAB_GEBUEHZIFFER",
        "dist_1_end_date",
        "eq_start_date",
    ],
    [
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_start_date",
    ],
    [
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "overlap_0_clean_DIAG_ZIFFER",
        "eq_end_date",
        "eq_start_date",
    ],
    [
        "overlap_0_TAB_DAT_LSTG_VON",
        "prop_0.99_clean_BEL_BETR_RECHNUNG",
        "overlap_0_clean_DIAG_ZIFFER",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    ],
    [
        "eq_BE_STRASSE_NR",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "overlap_0_clean_TAB_GEBUEHZIFFER",
    ],
    [
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "eq_pos_len",
        "eq_start_date",
    ],
    ["eq_clean_BEL_BETR_RECHNUNG", "eq_end_date", "eq_pos_len", "dist_1_start_date"],
    [
        "eq_BE_STRASSE_NR",
        "prop_0.99_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_start_date",
    ],
    [
        "prop_0.99_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_end_date",
        "eq_pos_len",
    ],
    [
        "eq_BE_STRASSE_NR",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "prop_0.95_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    ],
    [
        "eq_DocClass",
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "eq_start_date",
    ],
    [
        "eq_LAR",
        "overlap_0_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_start_date",
    ],
    [
        "overlap_0_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_BETR_RECHNUNG",
        "overlap_0_clean_DIAG_ZIFFER",
        "dist_1_end_date",
    ],
    ["eq_BE_NAME", "eq_clean_BEL_BETR_RECHNUNG", "dist_1_end_date", "eq_start_date"],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "eq_end_date",
        "eq_start_date",
    ],
    [
        "eq_BE_STRASSE_NR",
        "eq_TAB_BETR_POSITION",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    ],
    [
        "eq_BE_STRASSE_NR",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "eq_start_date",
    ],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "dist_2_clean_BEL_BETR_RECHNUNG",
        "overlap_0_clean_DIAG_ZIFFER",
        "eq_start_date",
    ],
    ["eq_clean_BEL_BETR_RECHNUNG", "eq_clean_BEL_DAT_AUSSTELLUNG", "eq_start_date"],
    [
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "eq_end_date",
        "dist_2_start_date",
    ],
    ["overlap_0_TAB_DAT_LSTG_VON", "eq_clean_BEL_BETR_RECHNUNG", "eq_end_date"],
    [
        "prop_0.99_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "overlap_0_clean_TAB_GEBUEHZIFFER",
        "eq_end_date",
    ],
    [
        "prop_0.95_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_end_date",
    ],
    [
        "overlap_0_clean_DIAG_ZIFFER",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_end_date",
        "eq_pos_len",
    ],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "overlap_0.5_clean_DIAG_ZIFFER",
        "eq_start_date",
    ],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "eq_pos_len",
    ],
    [
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "eq_clean_BEL_DAT_VERORDNUNG",
        "dist_1_end_date",
    ],
    ["eq_PKV_PK", "eq_clean_BEL_BETR_RECHNUNG", "dist_2_end_date", "eq_start_date"],
    [
        "eq_BE_NAME",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "prop_0.99_clean_BEL_BETR_RECHNUNG",
        "eq_start_date",
    ],
    [
        "eq_BE_STRASSE_NR",
        "overlap_0_TAB_BETR_POSITION",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "eq_start_date",
    ],
    [
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "dist_2_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_pos_len",
    ],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_pos_len",
    ],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_end_date",
        "eq_pos_len",
    ],
    [
        "eq_BE_NAME",
        "overlap_0.5_clean_DIAG_ZIFFER",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_end_date",
    ],
    [
        "eq_BE_NAME",
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0.5_clean_DIAG_ZIFFER",
        "eq_start_date",
    ],
    ["eq_clean_BEL_BETR_RECHNUNG", "eq_end_date", "dist_2_start_date"],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0.5_TAB_DAT_LSTG_VON",
        "overlap_0_clean_DIAG_ZIFFER",
        "eq_end_date",
    ],
    [
        "overlap_0_TAB_BETR_POSITION",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "eq_end_date",
        "eq_start_date",
    ],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "overlap_0_TAB_DAT_LSTG_VON",
        "prop_0.95_clean_BEL_BETR_RECHNUNG",
        "eq_start_date",
    ],
    [
        "eq_TAB_BETR_POSITION",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "overlap_0.5_clean_DIAG_ZIFFER",
        "dist_1_start_date",
    ],
    [
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "overlap_0.5_clean_DIAG_ZIFFER",
        "dist_1_start_date",
    ],
    [
        "eq_clean_BEL_BETR_RECHNUNG",
        "eq_clean_BEL_DAT_AUSSTELLUNG",
        "overlap_0_clean_DIAG_ZIFFER",
        "dist_1_start_date",
    ],
    ["eq_clean_BEL_BETR_RECHNUNG", "dist_2_end_date", "eq_start_date"],
    [
        "overlap_0.5_TAB_BETR_POSITION",
        "dist_2_clean_BEL_BETR_RECHNUNG",
        "overlap_0.5_clean_TAB_GEBUEHZIFFER",
        "eq_end_date",
    ],
    [
        "eq_DocClass",
        "dist_1_clean_BEL_BETR_RECHNUNG",
        "prop_0.99_clean_BEL_BETR_RECHNUNG",
        "eq_end_date",
    ],
    # [
    #   "overlap_0_TAB_BETR_POSITION",
    #   "overlap_0.5_TAB_DAT_LSTG_VON",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "eq_pos_len"
    # ],
    # [
    #   "overlap_0_TAB_BETR_POSITION",
    #   "overlap_0.5_clean_DIAG_ZIFFER",
    #   "eq_end_date",
    #   "eq_pos_len"
    # ],
    # [
    #   "eq_DocClass",
    #   "dist_1_clean_BEL_BETR_RECHNUNG",
    #   "prop_0.95_clean_BEL_BETR_RECHNUNG",
    #   "eq_start_date"
    # ],
    # [
    #   "overlap_0.5_TAB_BETR_POSITION",
    #   "overlap_0.5_TAB_DAT_LSTG_VON",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "eq_end_date"
    # ],
    # [
    #   "overlap_0.5_TAB_BETR_POSITION",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "eq_end_date",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_BE_NAME",
    #   "overlap_0.5_TAB_BETR_POSITION",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "eq_start_date"
    # ],
    # [
    #   "overlap_0.5_TAB_BETR_POSITION",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "overlap_0_TAB_BETR_POSITION",
    #   "overlap_0.5_TAB_DAT_LSTG_VON",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER"
    # ],
    # [
    #   "overlap_0.5_TAB_DAT_LSTG_VON",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "eq_end_date"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "overlap_0.5_TAB_DAT_LSTG_VON",
    #   "overlap_0.5_clean_DIAG_ZIFFER",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER"
    # ],
    # [
    #   "eq_TAB_BETR_POSITION",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_BE_NAME",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_BE_NAME",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "dist_2_end_date",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0.5_clean_DIAG_ZIFFER",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "overlap_0.5_TAB_DAT_LSTG_VON",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0.5_clean_DIAG_ZIFFER",
    #   "dist_2_end_date"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "overlap_0.5_clean_DIAG_ZIFFER",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "overlap_0_clean_DIAG_ZIFFER",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_TAB_BETR_POSITION",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER",
    #   "dist_1_end_date",
    #   "dist_2_start_date"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "dist_2_clean_BEL_BETR_RECHNUNG",
    #   "eq_end_date",
    #   "eq_start_date"
    # ],
    # [
    #   "overlap_0_TAB_BETR_POSITION",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "dist_2_end_date",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "dist_1_end_date",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "dist_2_end_date"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "eq_pos_len",
    #   "dist_2_start_date"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "dist_1_end_date"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_TAB_BETR_POSITION",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER",
    #   "eq_pos_len"
    # ],
    # [
    #   "overlap_0_TAB_BETR_POSITION",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER"
    # ],
    # [
    #   "eq_BE_NAME",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "eq_pos_len"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "eq_pos_len",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_DocClass",
    #   "dist_1_clean_BEL_BETR_RECHNUNG",
    #   "dist_1_start_date"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_DocClass",
    #   "eq_pos_len",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "eq_end_date",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_DocClass",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_clean_BEL_DAT_AUSSTELLUNG",
    #   "eq_end_date",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "overlap_0_TAB_BETR_POSITION",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER",
    #   "eq_start_date"
    # ],
    # [
    #   "overlap_0_TAB_BETR_POSITION",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER",
    #   "eq_end_date",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_BE_NAME",
    #   "eq_PKV_PK",
    #   "eq_end_date",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_TAB_BETR_POSITION",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER",
    #   "dist_2_end_date",
    #   "dist_2_start_date"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "dist_2_end_date",
    #   "dist_2_start_date"
    # ],
    # [
    #   "eq_DocClass",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "dist_2_end_date"
    # ],
    # [
    #   "eq_PKV_PK",
    #   "eq_clean_BEL_DAT_AUSSTELLUNG"
    # ],
    # [
    #   "overlap_0_TAB_BETR_POSITION",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_TAB_BETR_POSITION",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER"
    # ],
    # [
    #   "eq_DocClass",
    #   "overlap_0_TAB_BETR_POSITION",
    #   "eq_clean_BEL_BETR_RECHNUNG",
    #   "overlap_0_clean_TAB_GEBUEHZIFFER"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_DocClass",
    #   "eq_PKV_PK",
    #   "eq_clean_BEL_BETR_RECHNUNG"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_clean_BEL_BETR_RECHNUNG"
    # ],
    # [
    #   "eq_clean_BEL_BETR_RECHNUNG"
    # ],
    # [
    #   "eq_DocClass",
    #   "eq_end_date",
    #   "eq_start_date"
    # ],
    # [
    #   "eq_start_date"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_PKV_PK",
    #   "overlap_0.5_TAB_BETR_POSITION",
    #   "overlap_0.5_clean_TAB_GEBUEHZIFFER"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_DocClass",
    #   "eq_PKV_PK",
    #   "eq_pos_len"
    # ],
    # [
    #   "eq_BE_STRASSE_NR",
    #   "eq_DocClass",
    #   "dist_2_end_date",
    #   "dist_2_start_date"
    # ],
    # [
    #   "prop_0.95_clean_BEL_BETR_RECHNUNG"
    # ],
    # [
    #   "overlap_0_TAB_BETR_POSITION",
    #   "dist_2_end_date",
    #   "dist_2_start_date"
    # ]
]

DATA_PATH = Path(
    filepath_shared_folder
    / "dz/labelling/full_data_sets"
    / "25_12_25_2y_historic_all_labels_without_unclear__25_04_2026_.parquet"
)
OUTPUT_EXCEL = Path(
    filepath_shared_folder
    / "dz/labelling/labels/00025__rule_historic_safir_22_04_2026_2.xlsx"
)


BASE_FILTER = (
    ~OLD_DZ_RULE
    & (~pl.col("lar").str.starts_with("3") | ~pl.col("lar__2").str.starts_with("3"))
    & pl.col("lar").ne("141")
)

TOTAL_SAMPLE_SIZE = 500  # Gesamtanzahl an Paaren für manuelles Labeln

SEED = 42


HISTORIC_UNLABELED_FILTER = (
    pl.col("HUKIMPORTTIME").dt.year().eq(2025)
    & pl.col("HUKIMPORTTIME").dt.month().eq(11)
    & pl.col("HUKIMPORTTIME__2").ge(pl.datetime(2023, 11, 1))
    & pl.col("HUKIMPORTTIME__2").lt(pl.datetime(2025, 12, 1))
    & ~(
        pl.col("HUKIMPORTTIME__2").ge(pl.datetime(2025, 10, 1))
        & pl.col("HUKIMPORTTIME__2").lt(pl.datetime(2025, 12, 25))
    )
)


def fetch_vnr_mapping() -> pl.DataFrame:
    logging.info("Hole VNR-Mapping aus VW_MR_AP_VNR_SCHUTZ...")
    conn = get_engine(database="dprdwhm")
    df = pl.read_database(
        query="SELECT AGREEMENT_ID, P_VNR_PB AS VNR FROM PR.VW_MR_AP_VNR_SCHUTZ",
        connection=conn,
    )
    logging.info(f"VNR-Mapping geladen: {df.height} Einträge")
    return df


def _first_hit(rule_cols: list[str]) -> pl.Expr:
    """
    Returns a Series with the index of the FIRST True rule column per row, or -1.

    arg_max on booleans returns the LAST True — so we reverse and subtract.
    """
    n = len(rule_cols)
    # Reverse the list, find last True in reversed = first True in original
    reversed_cols = rule_cols[::-1]
    reversed_list = pl.concat_list([pl.col(c) for c in reversed_cols])
    any_hit = pl.concat_list([pl.col(c) for c in rule_cols]).list.any()

    # arg_max on reversed list gives index of first True in original
    first_in_reversed = reversed_list.list.arg_max()
    first_hit_idx = (pl.lit(n - 1) - first_in_reversed).cast(pl.Int32)

    return pl.when(any_hit).then(first_hit_idx).otherwise(pl.lit(-1).cast(pl.Int32))


def generate_sample() -> pl.DataFrame:
    logging.info("Lese Parquet-Datei...")
    data = (
        pl.read_parquet(DATA_PATH)
        .filter(BASE_FILTER & HISTORIC_UNLABELED_FILTER)
        .with_columns(
            pl.sum_horizontal(
                pl.col("^eq.*$"), pl.col("^overlap.*$"), pl.col("^dist.*$")
            ).alias("sum_sim")
        )
    )
    logging.info(f"Nach Filter: {data.height} Zeilen")

    rule_cols = [f"_rule_{i}" for i in range(len(GREEDY_RULES))]
    rule_exprs = []
    for i, rule in enumerate(GREEDY_RULES):
        mask = pl.lit(True)
        for col in rule:
            mask = mask & pl.col(col).cast(pl.Boolean).fill_null(False)
        rule_exprs.append(mask.alias(rule_cols[i]))

    data = data.with_columns(rule_exprs)

    data = data.with_columns(_first_hit(rule_cols).alias("stratum_k")).drop(rule_cols)

    data = data.filter(pl.col("stratum_k") >= 0)
    logging.info(f"Mit mindestens einem Regel-Hit: {data.height} Zeilen")

    strata = (
        (data.group_by("stratum_k").agg(pl.len().alias("N_k")).sort("stratum_k"))
        .with_columns(
            n_k_raw=(pl.col("N_k") / pl.col("N_k").sum() * TOTAL_SAMPLE_SIZE)
            .round()
            .cast(pl.Int32)
            .clip(1),
        )
        .with_columns(n_k=pl.min_horizontal("N_k", "n_k_raw"))
        .drop("n_k_raw")
    )

    logging.info("Stratum-Statistik:")
    for row in strata.rows(named=True):
        logging.info(f"  k={row['stratum_k']}, N_k={row['N_k']}, n_k={row['n_k']}")

    data = data.join(strata, on="stratum_k", how="left")

    sampled_dfs = []
    for row in strata.rows(named=True):
        k, n_k = row["stratum_k"], row["n_k"]
        subset = data.filter(pl.col("stratum_k") == k)
        sampled_dfs.append(subset.sample(n=n_k, seed=SEED, shuffle=True))

    result = pl.concat(sampled_dfs).sort("stratum_k")
    logging.info(f"Final: {result.height} Zeilen im Sample")
    return result


def main():
    logging.info("Starte stratifiziertes Sampling...")
    sampled = generate_sample()

    vnr_map = fetch_vnr_mapping()
    result = sampled.join(
        vnr_map,
        left_on=pl.col("agreement_id").cast(pl.String),
        right_on=pl.col("AGREEMENT_ID").cast(pl.String),
        how="left",
    )

    flatfiles = pl.scan_parquet(
        filepath_shared_folder / "export_xml_flatfiles/dkxml/*/flatfile.parquet"
    ).select("StackID", "ProcessID", "DocID", "SubDocID", "pglbea_lejhr", "pglbea_nr")

    identifiers = AggregationIdentifiers.dev()
    labels = collect_dz_labels(identifiers=identifiers, filter_relevant=False)

    export_df_joined = (
        result.lazy()
        .join(
            other=flatfiles,
            how="left",
            on=["StackID", "ProcessID", "DocID", "SubDocID"],
        )
        .join(
            other=flatfiles,
            how="left",
            left_on=["StackID__2", "ProcessID__2", "DocID__2", "SubDocID__2"],
            right_on=["StackID", "ProcessID", "DocID", "SubDocID"],
            suffix="__2",
        )
        .join(
            other=labels.lazy(),
            how="left",
            on=identifiers.document_pair,
        )
        .filter(pl.all_horizontal("^dz_interesting.*$").is_null())
        .sort("sum_sim", descending=True)
        .select(
            "StackID",
            "ProcessID",
            "DocID",
            "SubDocID",
            "agreement_id",
            "pglbea_id",
            "StackID__2",
            "ProcessID__2",
            "DocID__2",
            "SubDocID__2",
            "pglbea_id__2",
            "rebtr",
            "rebtr__2",
            "start_date",
            "start_date__2",
            "end_date",
            "end_date__2",
            "pglbea_lejhr",
            "pglbea_nr",
            "pglbea_lejhr__2",
            "pglbea_nr__2",
            "VNR",
            pl.lit(None).cast(pl.String).alias("label"),
        )
        .collect()
    )

    logging.info(f"Schreibe Excel: {OUTPUT_EXCEL}")
    export_df_joined.write_excel(
        OUTPUT_EXCEL,
        autofit=True,
    )

    logging.info(f"Excel: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
