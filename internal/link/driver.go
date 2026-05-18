// Package link spawns the system linker (ld) to produce the final
// executable. This is the single external-tool dependency in v0.0.
package link

import (
	"fmt"
	"os/exec"
)

// Link invokes `ld` with the supplied object files, producing outPath.
// It uses an x86-64 Linux dynamic-linker-free static link suitable for
// our minimal _start runtime.
func Link(outPath string, objs []string, extraArgs ...string) error {
	args := []string{"-o", outPath, "-static"}
	args = append(args, extraArgs...)
	args = append(args, objs...)
	cmd := exec.Command("ld", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ld failed: %v\n%s", err, string(out))
	}
	return nil
}
