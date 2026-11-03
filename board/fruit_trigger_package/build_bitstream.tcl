set root [file dirname [file normalize [info script]]]
if {![llength [get_projects -quiet]]} {
    open_project [file join $root build fruit_trigger fruit_trigger.xpr]
}
if {[get_property NAME [current_project]] ne "fruit_trigger"} {error "Open the generated fruit_trigger project first"}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Build failed; inspect impl_1 log"}
open_run impl_1
set bad [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 0]
if {[llength $bad]} {error "Negative timing slack; inspect timing before using bitstream"}
set out [file join $root output]
file mkdir $out
set bit [file join [get_property DIRECTORY [get_runs impl_1]] design_1_wrapper.bit]
set hwhs [glob -nocomplain [file join $root build fruit_trigger fruit_trigger.gen sources_1 bd design_1 hw_handoff *.hwh]]
if {![llength $hwhs]} {error "HWH not found; inspect generated BD hw_handoff folder"}
file copy -force $bit [file join $out fruit_trigger.bit]
file copy -force [lindex $hwhs 0] [file join $out fruit_trigger.hwh]
puts "READY: $out/fruit_trigger.bit and fruit_trigger.hwh"
