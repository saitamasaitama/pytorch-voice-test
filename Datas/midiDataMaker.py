# midoをpipからinstallして仕様。
import glob
import mido

midList = glob.glob('*.mid')
notesList = []
for mid in midList:
  f = mido.MidiFile(mid)

  notes = [message.note for message in f.tracks[0] if message.type == 'note_on']
  if len(notes) != 0:
    notesList.append(', '.join(map(str, notes)))
print(notesList)
with open('notes_data.txt', mode='w') as f2:
  f2.write('\n'.join(notesList))