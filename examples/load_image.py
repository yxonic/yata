from yata.loaders import DirectoryLoader
from yata.fields import Image

loader = DirectoryLoader('imgs', Image((200, 100)))
print(loader.keys)
print(loader.get(loader.keys[0]))

# loader.get('abc').file.show()
# print(next(loader.epoch(32))[1].file.shape)
