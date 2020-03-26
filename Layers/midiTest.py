import pretty_midi

pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=120) #pretty_midiオブジェクトを作ります
instrument = pretty_midi.Instrument(0) #instrumentはトラックみたいなものです。

note_number = pretty_midi.note_name_to_number('G4')
note = pretty_midi.Note(velocity=10, pitch=note_number, start=0, end=1) #noteはNoteOnEventとNoteOffEventに相当します。

instrument.notes.append(note)
pm.instruments.append(instrument)
pm.write('test.mid') #midiファイルを書き込みます。