// Copyright (C) 2002 Charless C. Fowlkes <fowlkes@eecs.berkeley.edu>
// Copyright (C) 2002 David R. Martin <dmartin@eecs.berkeley.edu>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
// 02111-1307, USA, or see http://www.gnu.org/copyleft/gpl.html.


#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "smatrix.hh"

SMatrix::SMatrix (int n, int* nz, int** col, double** values)
{
    this->n = n;
    this->nz = nz;
    this->col = col;
    this->values = values;
    int nnz = 0;
    for (int i = 0; i < n; i++)
      nnz += nz[i];
    printf("sparse matrix");//TODO what do with std spam?
    //Util::Message::debug(Util::String("creating sparse matrix with %d nonzero entries",nnz));
}

SMatrix::~SMatrix ()
{
  for (int i = 0; i < n; i++)
  {
    delete[] col[i];
    delete[] values[i];
  }
  delete col;
  delete values;
  delete nz;
}

void SMatrix::symmetrize()
{
  int* tail = new int[n];  
  memset(tail,0,n*sizeof(int));
  for (int r = 0; r < n; r++) 
  {
    int offset = 0;
    while ((offset < nz[r]) && (col[r][offset] < r+1))
    {
      offset++;
    }
    for (int i = offset; i < nz[r]; i++) 
    {
      int c = col[r][i];
      assert( col[c][tail[c]] == r ); 
      double v_rc = values[r][i];
      double v_cr = values[c][tail[c]];
      values[r][i] = 0.5*(v_rc+v_cr);
      values[c][tail[c]] = 0.5*(v_rc+v_cr);
      tail[c]++;
    }
  }  
}

