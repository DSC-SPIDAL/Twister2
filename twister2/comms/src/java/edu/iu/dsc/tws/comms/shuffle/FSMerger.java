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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import edu.iu.dsc.tws.comms.api.MessageType;

/**
 * Save the records to file system and retrieve them, this is just values, so no
 * sorting as in the keyed case
 */
public class FSMerger {
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
  private List<byte[]> bytesInMemory = new ArrayList<>();

  /**
   * The number of total bytes in each file part written to disk
   */
  private List<Integer> filePartBytes = new ArrayList<>();

  /**
   * Amount of bytes in the memory
   */
  private long numOfBytesInMemory = 0;

  /**
   * The type of value to be returned
   */
  private MessageType valueType;

  private Lock lock = new ReentrantLock();
  private Condition notFull = lock.newCondition();

  private enum FSStatus {
    WRITING,
    READING
  }

  private FSStatus status = FSStatus.WRITING;

  public FSMerger(int maxBytesInMemory, int maxRecsInMemory,
                  String dir, String opName, MessageType vType) {
    this.maxBytesToKeepInMemory = maxBytesInMemory;
    this.maxRecordsInMemory = maxRecsInMemory;
    this.folder = dir;
    this.operationName = opName;
    this.valueType = vType;
  }

  /**
   * Add the data to the file
   * @param data
   * @param length
   */
  public void add(byte[] data, int length) {
    if (status == FSStatus.READING) {
      throw new RuntimeException("Cannot add after switching to reading");
    }

    lock.lock();
    try {
      bytesInMemory.add(data);
      bytesLength.add(length);

      numOfBytesInMemory += length;
      if (numOfBytesInMemory > maxBytesToKeepInMemory
          || bytesInMemory.size() > maxRecordsInMemory) {
        notFull.signal();
      }
    } finally {
      lock.unlock();
    }
  }

  public void switchToReading() {
    status = FSStatus.READING;
  }

  /**
   * This method saves the data to file system
   */
  public void run() {
    lock.lock();
    try {
      // it is time to write
      if (numOfBytesInMemory > maxBytesToKeepInMemory
          || bytesInMemory.size() > maxRecordsInMemory) {
        // save the bytes to disk
        FileLoader.saveObjects(bytesInMemory, bytesLength,
            numOfBytesInMemory, getSaveFileName(noOfFileWritten));
        // save the sizes to disk
        FileLoader.saveSizes(bytesLength, getSizesFileName(noOfFileWritten));

        bytesInMemory.clear();
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
  public Iterator<byte[]> readIterator() {
    // lets start with first file
    return new FSIterator();
  }

  private class FSIterator implements Iterator<byte[]> {
    // the current file index
    private int currentFileIndex = 0;
    // Index of the current file
    private int currentIndex = 0;
    // the iterator for list of bytes in memory
    private Iterator<byte[]> it;
    // the current file part opened
    private OpenFile openFilePartBytes;
    // the current sizes file part
    private OpenFile openFilePartSizes;
    private List<Integer> dataSizes;

    FSIterator() {
      it = bytesInMemory.iterator();
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
        if (currentIndex < dataSizes.size()) {
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
      openFilePartBytes = FileLoader.openSavedPart(getSaveFileName(currentFileIndex));
      // read the complete bytes
      openFilePartSizes = FileLoader.openSavedPart(getSizesFileName(currentFileIndex));
      dataSizes = new ArrayList<>();
      int noOfData = 0;
      try {
        long size = openFilePartSizes.getRwChannel().size();
        if (size == 0) {
          throw new RuntimeException("File part with no data: "
              + getSizesFileName(currentIndex));
        }
        noOfData = (int) (size / 4);
      } catch (IOException e) {
        throw new RuntimeException("Cannot access file part: "
            + getSizesFileName(currentIndex));
      }
      // now lets read the sizes
      for (int i = 0; i < noOfData; i++) {
        dataSizes.add(openFilePartSizes.getByteBuffer().getInt());
      }
      currentFileIndex++;
      currentIndex = 0;
    }

    @Override
    public byte[] next() {
      // we are reading from in memory
      if (currentFileIndex == 0) {
        return it.next();
      }

      if (currentFileIndex > 0) {
        int size = dataSizes.get(currentIndex);
        byte[] data = new byte[size];
        openFilePartBytes.getByteBuffer().get(data);
        return data;
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

  /**
   * Get the name of the sizes file name
   * @param filePart file part index
   * @return filename
   */
  private String getSizesFileName(int filePart) {
    return folder + "/" + operationName + "/part_sizes_" + filePart;
  }
}
