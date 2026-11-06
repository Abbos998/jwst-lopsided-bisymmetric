# Superseded results (disk radius read in 0.05" pixels)

These robustness tests were run before it was found that DISK_RADIUS_pix is given
in 0.04" pixels (logs/author_machine_rd_units_check.log). With the wrong scale the
annulus was 0.63-1.88 r_d and the injected models were 25% too large. They are kept
only for reference; all results in the paper come from run_rd_fix.sh (r_ref_scale 0.04).
