//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.checkpointing.stores;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.NoSuchFileException;
import java.nio.file.OpenOption;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.checkpointing.StateStore;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.FileSystemContext;

public class LocalFileStateStore implements StateStore {

  private static final Logger LOG = Logger.getLogger(LocalFileStateStore.class.getName());

  private File rootFolder;

  @Override
  public void init(Config config, String... path) {
    String finalPath =
        String.join(File.separator, FileSystemContext.persistentStorageRoot(config),
            String.join(File.separator, path));
    this.rootFolder = new File(finalPath);
    LOG.info("Snapshot Store path : " + this.rootFolder.getAbsolutePath());
    if (!this.rootFolder.exists()) {
      boolean mkdirs = this.rootFolder.mkdirs();
      if (!mkdirs) {
        throw new RuntimeException("Couldn't create directories "
            + "for local file state store.");
      }
    }
  }

  private FileChannel getChannelForKey(String key, OpenOption... openOptions) throws IOException {
    return FileChannel.open(Paths.get(rootFolder.getAbsolutePath(), key), openOptions);
  }

  @Override
  public void put(String key, byte[] data) throws IOException {
    FileChannel fileChannel = this.getChannelForKey(key,
        StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    fileChannel.write(ByteBuffer.wrap(data));
    fileChannel.close();
  }

  @Override
  public byte[] get(String key) throws IOException {
    try {
      FileChannel fileChannel = this.getChannelForKey(key, StandardOpenOption.READ);
      int size = (int) fileChannel.size(); // assume < 2GB
      ByteBuffer allocate = ByteBuffer.allocate(size);
      fileChannel.read(allocate);
      return allocate.array();
    } catch (NoSuchFileException nex) {
      return null;
    }
  }

  @Override
  public boolean contains(String key) throws IOException {
    try (FileChannel channel = getChannelForKey(key, StandardOpenOption.READ)) {
      return true;
    } catch (NoSuchFileException nex) {
      return false;
    }
  }
}
