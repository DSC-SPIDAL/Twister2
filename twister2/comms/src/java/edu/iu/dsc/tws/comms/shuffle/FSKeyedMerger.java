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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Logger;

import edu.iu.dsc.tws.comms.api.MessageType;
import edu.iu.dsc.tws.data.utils.KryoMemorySerializer;

public class FSKeyedMerger {
  private static final Logger LOG = Logger.getLogger(FSKeyedMerger.class.getName());

  /**
   * Maximum bytes to keep in memory
   */
  private int maxBytesToKeepInMemory;

  /**
   * Maximum number of records in memory. We will choose lesser of two maxes to write to disk
   */
  private int maxRecordsInMemory;

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
  private List<KeyValue> recordsInMemory = new ArrayList<>();

  /**
   * The deserialized objects in memory
   */
  private List<KeyValue> objectsInMemory = new ArrayList<>();

  /**
   * The number of total bytes in each file part written to disk
   */
  private List<Integer> filePartBytes = new ArrayList<>();

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

  /**
   * The key comparator used for comparing keys
   */
  private Comparator<Object> keyComparator;

  private Lock lock = new ReentrantLock();
  private Condition notFull = lock.newCondition();

  /**
   * The kryo serializer
   */
  private KryoMemorySerializer kryoSerializer;

  private enum FSStatus {
    WRITING,
    READING
  }

  private FSStatus status = FSStatus.WRITING;

  public FSKeyedMerger(int maxBytesInMemory, int maxRecsInMemory,
                       String dir, String opName, MessageType kType,
                       MessageType dType, Comparator<Object> kComparator) {
    this.maxBytesToKeepInMemory = maxBytesInMemory;
    this.maxRecordsInMemory = maxRecsInMemory;
    this.folder = dir;
    this.operationName = opName;
    this.keyType = kType;
    this.dataType = dType;
    this.keyComparator = kComparator;
    this.kryoSerializer = new KryoMemorySerializer();
  }

  /**
   * Add the data to the file
   * @param data
   * @param length
   */
  public void add(Object key, byte[] data, int length) {
    if (status == FSStatus.READING) {
      throw new RuntimeException("Cannot add after switching to reading");
    }

    lock.lock();
    try {
      recordsInMemory.add(new KeyValue(key, data));
      bytesLength.add(length);

      numOfBytesInMemory += length;
      if (numOfBytesInMemory > maxBytesToKeepInMemory
          || recordsInMemory.size() > maxRecordsInMemory) {
        notFull.signal();
      }
    } finally {
      lock.unlock();
    }
  }

  public void switchToReading() {
    status = FSStatus.READING;
    // lets convert the in-memory data to objects
    deserializeObjects();
  }

  private void deserializeObjects() {
    for (int i = 0; i < recordsInMemory.size(); i++) {
      KeyValue kv = recordsInMemory.get(i);
      Object o = kryoSerializer.deserialize((byte[]) kv.getValue());
      objectsInMemory.add(new KeyValue(kv.getKey(), o));
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
        int totalSize = FileLoader.saveKeyValues(recordsInMemory, bytesLength,
            numOfBytesInMemory, getSaveFileName(noOfFileWritten), keyType, kryoSerializer);
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
  public Iterator<KeyValue> readIterator() {
    // lets start with first file
    return new FSIterator();
  }

  private class FSIterator implements Iterator<KeyValue> {
    // the current file index
    private int currentFileIndex = 0;
    // Index of the current file
    private int currentIndex = 0;
    // the iterator for list of bytes in memory
    private Iterator<KeyValue> it;
    // the current values
    private List<KeyValue> openValue;

    FSIterator() {
      it = objectsInMemory.iterator();
    }

    @Override
    public boolean hasNext() {
      // we are reading from in memory
      boolean next;
      if (currentFileIndex == 0) {
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

      if (currentFileIndex > 0) {
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


    private void openFilePart() {
      // lets read the bytes from the file
      openValue = FileLoader.readFile(getSaveFileName(currentFileIndex), keyType,
          dataType, kryoSerializer);
      currentFileIndex++;
      currentIndex = 0;
    }

    @Override
    public KeyValue next() {
      // we are reading from in memory
      if (currentFileIndex == 0) {
        return it.next();
      }

      if (currentFileIndex > 0) {
        KeyValue kv = openValue.get(currentIndex);
        currentIndex++;
        return kv;
      }

      return null;
    }
  }

  /**
   * Get the file name to save the current part
   * @param filePart file part index
   * @return the save file name
   */
  private String getSaveFileName(int filePart) {
    return folder + "/" + operationName + "/part_" + filePart;
  }
}
