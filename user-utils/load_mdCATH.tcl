proc load_mdCATH {fn temperature replica} {

    set pid [pid]

    # Try to execute h5ls and handle errors
    set status [catch {exec h5ls $fn} tmp]
    if {$status} {
        return -code error "Error executing h5ls on file $fn: $tmp"
    }
    set code [lindex $tmp 0]

    set tmpdir $::env(TMPDIR)
    set pdbname $tmpdir/loadmdcath.$pid.pdb

    # Handle potential errors from h5dump
    if {[catch {exec h5dump -b -o $pdbname -d /$code/pdbProteinAtoms $fn} result]} {
        return -code error "Error dumping pdbProteinAtoms from $fn: $result"
    }

    # Load the molecular data
    mol new $pdbname
    file delete $pdbname

    set N [molinfo top get numatoms]
    if {$N == 0} {
        return -code error "No atoms found in the molecule loaded from $pdbname"
    }

    animate delete all
    set cbin $tmpdir/loadmdcath.$pid.coords.bin
    if {[catch {exec h5dump -b -o $cbin -d /$code/sims${temperature}K/$replica/coords $fn} result]} {
        return -code error "Error dumping coords from $fn: $result"
    }

    # Handle file opening and binary data reading
    if {[catch {open $cbin r} fp msg]} {
        return -code error "Error opening coordinates file $cbin: $msg"
    }
    fconfigure $fp -translation binary
    if {[catch {read $fp} cdat]} {
        close $fp
        return -code error "Error reading data from coordinates file $cbin"
    }
    close $fp
    file delete $cbin

    # Binary data processing
    set M [binary scan $cdat f* dat]
    if {$M == 0} {
        return -code error "Failed to scan binary data from $cbin"
    }

    set L [llength $dat]
    set T [expr {$L/$N/3.0}]
    set N3 [expr {$N * 3}]
    set N3m1 [expr {$N3-1}]

    puts "Assuming $T frames"

    set a [atomselect top all]

    for {set t 0} {$t<$T} {incr t} {
        animate dup top
        set xyz {}
        set fcoor [lrange $dat 0 $N3m1]
        set dat [lreplace $dat 0 $N3m1]
        foreach {x y z} $fcoor {
            lappend xyz [list $x $y $z]
        }
        $a set {x y z} $xyz
        $a update
    }
    $a delete
    mol rename top "mdCATH: $code $temperatureK $replica"

    # TODO set box
}
