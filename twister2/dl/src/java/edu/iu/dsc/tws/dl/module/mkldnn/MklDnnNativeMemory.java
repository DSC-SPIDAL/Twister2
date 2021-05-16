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
package edu.iu.dsc.tws.dl.module.mkldnn;

@SuppressWarnings({"LocalVariableName", "ParameterName", "MemberName"})
public abstract class MklDnnNativeMemory implements Releasable {
  protected long __ptr;
  protected MemoryOwner owner;
  private long UNDEFINED = -1L;
  private long ERROR = 0L;
  private boolean isUndefOrError = __ptr == UNDEFINED || __ptr == ERROR;

  public MklDnnNativeMemory(long __ptr, MemoryOwner owner) {
    this.__ptr = __ptr;
    this.owner = owner;
    owner.registerResource(this);
  }

  @Override
  public void release() {
    if (!isUndefOrError) {
      doRelease();
      reset();
    }
  }

  public abstract void doRelease();

  public void reset() {
    __ptr = ERROR;
  }

  public long getPtr() {
    return __ptr;
  }
}
