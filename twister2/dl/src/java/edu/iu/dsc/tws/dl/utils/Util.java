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
package edu.iu.dsc.tws.dl.utils;

import edu.iu.dsc.tws.dl.data.Table;

public final class Util {

  private Util() {
  }

  public static void require(boolean satisfied, String message) {


    if (!satisfied) {
      throw new IllegalStateException(message);
    }
  }

  public static void require(boolean satisfied) {
    if (!satisfied) {
      throw new IllegalStateException("Requirement not met");
    }
  }

  public static int getHashCode(Object o){
    if(o == null){
      return 0;
    }else{
      return o.hashCode();
    }
  }

  public <K> K allocate(){
    if(K instanceof Table){

    }
  }
}
