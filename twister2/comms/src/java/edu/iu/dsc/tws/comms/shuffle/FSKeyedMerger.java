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
package edu.iu.dsc.tws.comms.shuffle;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.io.FileUtils;

import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.util.KryoSerializer;

/**
 * Un sorted merger
 * <p>
 * This merger can't output data in the expected format Iterator<Tuple<Key,Iterator>>
 */
@SuppressWarnings({"unchecked", "rawtypes"})
public class FSKeyedMerger implements Shuffle {
  private static final Logger LOG = Logger.getLogger(FSKeyedMerger.class.getName());

  /**
   * Maximum bytes to keep in memory
   */
  private long maxBytesToKeepInMemory;

  /**
   * Maximum number of records in memory. We will choose lesser of two maxes to write to disk
   */
  private long maxRecordsInMemory;

  /**
   * The base folder to work on
   */
  private String folder;

  /**
   * Operation name
   */
  private String operationName;

  /**
   * No of files written to the disk so far
   * The files are started from 0 and go up to this amount
   */
  private int noOfFileWritten = 0;

  /**
   * The size of the records in memory
   */
  private List<Integer> bytesLength = new ArrayList<>();

  /**
   * List of bytes in the memory so far
   */
  private List<Tuple> recordsInMemory = new ArrayList<>();

  /**
   * The deserialized objects in memory
   */
  private List<Tuple> objectsInMemory = new ArrayList<>();

  /**
   * The number of total bytes in each file part written to disk
   */
  private List<Long> filePartBytes = new ArrayList<>();

  /**
   * Amount of bytes in the memory
   */
  private long numOfBytesInMemory = 0;

  /**
   * The type of the key used
   */
  private MessageType keyType;

  /**
   * The data type to be returned, by default it is byte array
   */
  private MessageType dataType;

  private Lock lock = new ReentrantLock();

  /**
   * The kryo serializer
   */
  private KryoSerializer kryoSerializer;

  private enum FSStatus {
    WRITING,
    READING
  }

  private FSStatus status = FSStatus.WRITING;

  public FSKeyedMerger(long maxBytesInMemory, long maxRecsInMemory,
                       String dir, String opName, MessageType kType,
                       MessageType dType) {
    this.maxBytesToKeepInMemory = maxBytesInMemory;
    this.maxRecordsInMemory = maxRecsInMemory;
    this.folder = dir;
    this.operationName = opName;
    this.keyType = kType;
    this.dataType = dType;
    this.kryoSerializer = new KryoSerializer();
  }

  /**
   * Add the data to the file
   */
  public void add(Object key, byte[] data, int length) {
    if (status == FSStatus.READING) {
      throw new RuntimeException("Cannot add after switching to reading");
    }

    lock.lock();
    try {
      recordsInMemory.add(new Tuple(key, data));
      bytesLength.add(length);

      numOfBytesInMemory += length;
    } finally {
      lock.unlock();
    }
  }

  public void switchToReading() {
    lock.lock();
    try {
      status = FSStatus.READING;
      // lets convert the in-memory data to objects
      deserializeObjects();
    } finally {
      lock.unlock();
    }
  }

  private void deserializeObjects() {
    for (int i = 0; i < recordsInMemory.size(); i++) {
      Tuple kv = recordsInMemory.get(i);
      Object o = dataType.getDataPacker().unpackFromByteArray((byte[]) kv.getValue());
      objectsInMemory.add(new Tuple(kv.getKey(), o));
    }
  }

  /**
   * This method saves the data to file system
   */
  public void run() {
    lock.lock();
    try {
      // it is time to write
      if (numOfBytesInMemory > maxBytesToKeepInMemory
          || recordsInMemory.size() > maxRecordsInMemory) {
        // save the bytes to disk
        long totalSize = FileLoader.saveKeyValues(recordsInMemory, bytesLength,
            numOfBytesInMemory, getSaveFileName(noOfFileWritten), keyType);
        filePartBytes.add(totalSize);

        recordsInMemory.clear();
        bytesLength.clear();
        noOfFileWritten++;
        numOfBytesInMemory = 0;
      }
    } finally {
      lock.unlock();
    }
  }

  /**
   * This method gives the values
   */
  public ResettableIterator<Object> readIterator() {
    // lets start with first file
    return new FSIterator();
  }

  private class FSIterator implements ResettableIterator<Object> {
    // the current file index
    private int currentFileIndex = -1;
    // Index of the current file
    private int currentIndex = 0;
    // the iterator for list of bytes in memory
    private Iterator<Tuple> it;
    // the current values
    private List<Tuple> openValue;

    private Tuple nextTuple;

    FSIterator() {
      it = objectsInMemory.iterator();
      this.createNextTuple();
    }

    private boolean nextTupleAvailable() {
      // we are reading from in memory
      boolean next;
      if (currentFileIndex == -1) {
        next = it.hasNext();
        if (!next) {
          // we need to open the first file part
          if (noOfFileWritten > 0) {
            // we will read the opened one next
            openFilePart();
          } else {
            // no file parts written, end of iteration
            return false;
          }
        } else {
          return true;
        }
      }

      if (currentFileIndex >= 0) {
        if (currentIndex < openValue.size()) {
          return true;
        } else {
          if (currentFileIndex < noOfFileWritten - 1) {
            openFilePart();
            return true;
          } else {
            return false;
          }
        }
      }
      return false;
    }

    private void createNextTuple() {
      nextTuple = null;
      if (nextTupleAvailable()) {
        if (currentFileIndex == -1) {
          nextTuple = it.next();
        }

        if (currentFileIndex >= 0) {
          nextTuple = openValue.get(currentIndex);
          currentIndex++;
        }
      }
    }

    @Override
    public boolean hasNext() {
      return nextTuple != null;
    }


    private void openFilePart() {
      // lets read the bytes from the file
      currentFileIndex++;
      openValue = FileLoader.readFile(getSaveFileName(currentFileIndex), keyType,
          dataType, kryoSerializer);
      currentIndex = 0;
    }

    @Override
    public Tuple next() {
      Tuple next = nextTuple;
      createNextTuple();
      return next;
    }

    @Override
    public void reset() {
      it = objectsInMemory.iterator();
      currentFileIndex = -1;
      currentIndex = 0;
      openValue = null;
      this.createNextTuple();
    }
  }

  /**
   * Cleanup the directories
   */
  public void clean() {
    File file = new File(getSaveFolderName());
    try {
      FileUtils.cleanDirectory(file);
    } catch (IOException e) {
      LOG.log(Level.SEVERE, "Failed to clear directory: " + file, e);
    }
  }

  /**
   * Get the file name to save the current part
   *
   * @return the save file name
   */
  private String getSaveFolderName() {
    return folder + "/" + operationName;
  }


  /**
   * Get the file name to save the current part
   *
   * @param filePart file part index
   * @return the save file name
   */
  private String getSaveFileName(int filePart) {
    return folder + "/" + operationName + "/part_" + filePart;
  }
}
