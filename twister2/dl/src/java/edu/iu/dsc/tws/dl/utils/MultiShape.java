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

public class MultiShape implements Shape{

  private Shape[] values;

  public MultiShape(Shape[] values) {
    this.values = values;
  }

  @Override
  public Shape[] toMulti() {
    return values;
  }

  @Override
  public Shape copyAndUpdate(int dim, Shape v) {
    Shape[] updated = values.clone();
    updated[getDim(dim, values.length)] = v;
    return new MultiShape(updated);
  }

  @Override
  public int hashCode() {
    int prime = 31;
    int result = 1;
    result = prime * values.hashCode();
    return result;
  }

  @Override
  public boolean equals(Object o) {
    if(o instanceof MultiShape && this.hashCode() == o.hashCode()){
      return true;
    }else{
      return false;
    }
  }
}
