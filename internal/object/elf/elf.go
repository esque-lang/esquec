// Package elf is a minimal ELF64 little-endian relocatable object writer
// for x86-64 Linux. It supports just enough to produce a `.o` that the
// system linker can consume: .text, .symtab, .strtab, .shstrtab, plus
// .rela.text when relocations are present, and an empty .note.GNU-stack
// to mark the stack non-executable.
package elf

import (
	"bytes"
	"encoding/binary"
)

// ELF64 constants we use.
const (
	EI_NIDENT = 16

	ELFCLASS64    = 2
	ELFDATA2LSB   = 1
	EV_CURRENT    = 1
	ELFOSABI_SYSV = 0

	ET_REL    = 1
	EM_X86_64 = 62

	SHT_NULL     = 0
	SHT_PROGBITS = 1
	SHT_SYMTAB   = 2
	SHT_STRTAB   = 3
	SHT_RELA     = 4
	SHT_NOBITS   = 8

	SHF_WRITE     = 0x1
	SHF_ALLOC     = 0x2
	SHF_EXECINSTR = 0x4
	SHF_INFO_LINK = 0x40

	STB_LOCAL  = 0
	STB_GLOBAL = 1

	STT_NOTYPE  = 0
	STT_OBJECT  = 1
	STT_FUNC    = 2
	STT_SECTION = 3

	SHN_UNDEF = 0

	R_X86_64_PC32  = 2
	R_X86_64_PLT32 = 4
)

// Symbol describes a defined or undefined symbol in the object.
type Symbol struct {
	Name    string
	Section string // section name; "" => undefined (extern)
	Value   uint64 // offset within Section
	Size    uint64
	Type    uint8
	Bind    uint8
}

// Reloc is a relocation in section ApplyTo.
type Reloc struct {
	ApplyTo string // section being relocated, typically ".text"
	Offset  uint64
	Sym     string
	Type    uint32 // R_X86_64_*
	Addend  int64
}

// Section is a raw byte section.
type Section struct {
	Name  string
	Type  uint32
	Flags uint64
	Data  []byte
	Align uint64
}

// Object is the in-memory representation of the .o we will write.
type Object struct {
	Sections []Section
	Symbols  []Symbol
	Relocs   []Reloc
}

// Add appends a section.
func (o *Object) Add(s Section) { o.Sections = append(o.Sections, s) }

type stringTable struct {
	buf  bytes.Buffer
	idxs map[string]uint32
}

func newStringTable() *stringTable {
	st := &stringTable{idxs: map[string]uint32{}}
	st.buf.WriteByte(0)
	st.idxs[""] = 0
	return st
}

func (s *stringTable) add(str string) uint32 {
	if i, ok := s.idxs[str]; ok {
		return i
	}
	i := uint32(s.buf.Len())
	s.buf.WriteString(str)
	s.buf.WriteByte(0)
	s.idxs[str] = i
	return i
}

// laidSection is a section after layout decisions are made.
type laidSection struct {
	name    string
	typ     uint32
	flags   uint64
	data    []byte
	align   uint64
	link    uint32
	info    uint32
	entsize uint64
	fileOff uint64
}

// Write serialises the object as an ELF64 little-endian relocatable file.
func (o *Object) Write() ([]byte, error) {
	// 1) Build the full section list (including .symtab, .strtab,
	//    one .rela.<sec> per relocated section, and .shstrtab) BEFORE
	//    laying out file offsets.

	var laid []laidSection
	// index 0: NULL section
	laid = append(laid, laidSection{})

	userSecIdx := map[string]int{}
	for _, s := range o.Sections {
		userSecIdx[s.Name] = len(laid)
		al := s.Align
		if al == 0 {
			al = 1
		}
		laid = append(laid, laidSection{
			name: s.Name, typ: s.Type, flags: s.Flags, data: s.Data, align: al,
		})
	}

	// 2) Build symbol table.
	strtab := newStringTable()
	type elfSym struct {
		name  uint32
		info  uint8
		other uint8
		shndx uint16
		value uint64
		size  uint64
	}
	var syms []elfSym
	syms = append(syms, elfSym{}) // index 0 undefined
	symIdx := map[string]int{"": 0}

	addSym := func(s Symbol) {
		var shndx uint16
		if s.Section == "" {
			shndx = SHN_UNDEF
		} else if i, ok := userSecIdx[s.Section]; ok {
			shndx = uint16(i)
		} else {
			shndx = SHN_UNDEF
		}
		syms = append(syms, elfSym{
			name:  strtab.add(s.Name),
			info:  (s.Bind << 4) | (s.Type & 0xF),
			shndx: shndx,
			value: s.Value,
			size:  s.Size,
		})
		symIdx[s.Name] = len(syms) - 1
	}
	for _, s := range o.Symbols {
		if s.Bind == STB_LOCAL {
			addSym(s)
		}
	}
	firstNonLocal := uint32(len(syms))
	for _, s := range o.Symbols {
		if s.Bind != STB_LOCAL {
			addSym(s)
		}
	}

	const symEntSize = 24
	var symBuf bytes.Buffer
	for _, s := range syms {
		var b [symEntSize]byte
		binary.LittleEndian.PutUint32(b[0:4], s.name)
		b[4] = s.info
		b[5] = s.other
		binary.LittleEndian.PutUint16(b[6:8], s.shndx)
		binary.LittleEndian.PutUint64(b[8:16], s.value)
		binary.LittleEndian.PutUint64(b[16:24], s.size)
		symBuf.Write(b[:])
	}

	symtabIdx := len(laid)
	laid = append(laid, laidSection{
		name: ".symtab", typ: SHT_SYMTAB, data: symBuf.Bytes(), align: 8,
		entsize: symEntSize,
	})
	strtabIdx := len(laid)
	laid = append(laid, laidSection{
		name: ".strtab", typ: SHT_STRTAB, data: strtab.buf.Bytes(), align: 1,
	})
	laid[symtabIdx].link = uint32(strtabIdx)
	laid[symtabIdx].info = firstNonLocal

	// 3) Relocation sections.
	relGroups := map[string][]Reloc{}
	var relGroupOrder []string
	for _, r := range o.Relocs {
		if _, ok := relGroups[r.ApplyTo]; !ok {
			relGroupOrder = append(relGroupOrder, r.ApplyTo)
		}
		relGroups[r.ApplyTo] = append(relGroups[r.ApplyTo], r)
	}
	const relaEntSize = 24
	for _, sec := range relGroupOrder {
		targetIdx, ok := userSecIdx[sec]
		if !ok {
			continue
		}
		var buf bytes.Buffer
		for _, r := range relGroups[sec] {
			si, ok := symIdx[r.Sym]
			if !ok {
				si = 0
			}
			var b [relaEntSize]byte
			binary.LittleEndian.PutUint64(b[0:8], r.Offset)
			info := (uint64(si) << 32) | uint64(r.Type)
			binary.LittleEndian.PutUint64(b[8:16], info)
			binary.LittleEndian.PutUint64(b[16:24], uint64(r.Addend))
			buf.Write(b[:])
		}
		laid = append(laid, laidSection{
			name: ".rela" + sec, typ: SHT_RELA, flags: SHF_INFO_LINK,
			data: buf.Bytes(), align: 8,
			link: uint32(symtabIdx), info: uint32(targetIdx),
			entsize: relaEntSize,
		})
	}

	// 4) Build .shstrtab now that all section names are known.
	shstrtab := newStringTable()
	for i := range laid {
		shstrtab.add(laid[i].name)
	}
	shstrtab.add(".shstrtab")
	shstrtabIdx := len(laid)
	laid = append(laid, laidSection{
		name: ".shstrtab", typ: SHT_STRTAB, data: shstrtab.buf.Bytes(), align: 1,
	})

	// 5) Lay out file offsets.
	const ehSize = 64
	const shEntSize = 64
	off := uint64(ehSize)
	for i := range laid {
		s := &laid[i]
		if s.typ == SHT_NULL || s.typ == SHT_NOBITS {
			continue
		}
		if s.align > 1 {
			rem := off % s.align
			if rem != 0 {
				off += s.align - rem
			}
		}
		s.fileOff = off
		off += uint64(len(s.data))
	}
	if off%8 != 0 {
		off += 8 - off%8
	}
	shoff := off

	var out bytes.Buffer
	out.Grow(int(shoff) + len(laid)*shEntSize)

	// ELF header
	var eh [ehSize]byte
	eh[0] = 0x7f
	eh[1] = 'E'
	eh[2] = 'L'
	eh[3] = 'F'
	eh[4] = ELFCLASS64
	eh[5] = ELFDATA2LSB
	eh[6] = EV_CURRENT
	eh[7] = ELFOSABI_SYSV
	binary.LittleEndian.PutUint16(eh[16:18], ET_REL)
	binary.LittleEndian.PutUint16(eh[18:20], EM_X86_64)
	binary.LittleEndian.PutUint32(eh[20:24], EV_CURRENT)
	binary.LittleEndian.PutUint64(eh[40:48], shoff)
	binary.LittleEndian.PutUint16(eh[52:54], ehSize)
	binary.LittleEndian.PutUint16(eh[54:56], 0)
	binary.LittleEndian.PutUint16(eh[56:58], 0)
	binary.LittleEndian.PutUint16(eh[58:60], shEntSize)
	binary.LittleEndian.PutUint16(eh[60:62], uint16(len(laid)))
	binary.LittleEndian.PutUint16(eh[62:64], uint16(shstrtabIdx))
	out.Write(eh[:])

	// Section data
	for i := range laid {
		s := &laid[i]
		if s.typ == SHT_NULL || s.typ == SHT_NOBITS {
			continue
		}
		for uint64(out.Len()) < s.fileOff {
			out.WriteByte(0)
		}
		out.Write(s.data)
	}
	for uint64(out.Len()) < shoff {
		out.WriteByte(0)
	}

	// Section header table.
	for i := range laid {
		s := &laid[i]
		var sh [shEntSize]byte
		nameOff := uint32(0)
		if s.name != "" {
			nameOff = shstrtab.add(s.name) // already added; this is just lookup
		}
		binary.LittleEndian.PutUint32(sh[0:4], nameOff)
		binary.LittleEndian.PutUint32(sh[4:8], s.typ)
		binary.LittleEndian.PutUint64(sh[8:16], s.flags)
		binary.LittleEndian.PutUint64(sh[16:24], 0)
		binary.LittleEndian.PutUint64(sh[24:32], s.fileOff)
		binary.LittleEndian.PutUint64(sh[32:40], uint64(len(s.data)))
		binary.LittleEndian.PutUint32(sh[40:44], s.link)
		binary.LittleEndian.PutUint32(sh[44:48], s.info)
		al := s.align
		if al == 0 {
			al = 1
		}
		binary.LittleEndian.PutUint64(sh[48:56], al)
		binary.LittleEndian.PutUint64(sh[56:64], s.entsize)
		out.Write(sh[:])
	}

	return out.Bytes(), nil
}
